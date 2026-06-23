"""Semantic FAQ cache — ChromaDB + SentenceBERT + adaptive threshold."""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class FAQCache:
    """Semantically caches question-answer pairs.

    When a new question arrives:
    1. Embed it
    2. Search for similar cached questions
    3. If similarity > threshold → return cached answer (HIT)
    4. Otherwise → return None (MISS), caller should ask LLM then call .store()

    Threshold auto-tunes if ``adaptive_threshold`` is enabled.
    """

    def __init__(
        self,
        data_dir: str = "./memcord_data",
        model_name: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.80,
        adaptive_threshold: bool = True,
        target_hit_rate: float = 0.75,
        adjustment_rate: float = 0.005,
    ):
        os.makedirs(data_dir, exist_ok=True)

        # Embedding model
        self._model = SentenceTransformer(model_name)

        # ChromaDB
        self._chroma = chromadb.PersistentClient(
            path=str(Path(data_dir) / "chroma"),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._chroma.get_or_create_collection(
            name="faqs", metadata={"hnsw:space": "cosine"}
        )

        # Threshold
        self.threshold = similarity_threshold
        self.adaptive = adaptive_threshold
        self.target_hit_rate = target_hit_rate
        self.adjustment_rate = adjustment_rate

        # Stats
        self._total = 0
        self._hits = 0
        self._stats_path = Path(data_dir) / "stats.json"
        self._load_stats()

    # ── public API ──────────────────────────────────────────

    def check(self, question: str) -> tuple[str | None, float]:
        """Check if a similar question is cached.

        Returns (answer, similarity_score) or (None, 0.0) on miss.
        """
        self._total += 1

        if self._collection.count() == 0:
            self._save_stats()
            return None, 0.0

        embedding = self._model.encode([question])[0].tolist()
        results = self._collection.query(
            query_embeddings=[embedding], n_results=1, include=["documents", "distances"]
        )

        if not results["ids"][0]:
            self._save_stats()
            return None, 0.0

        distance = results["distances"][0][0]
        similarity = 1.0 - distance  # cosine → similarity

        if similarity >= self.threshold:
            self._hits += 1
            # Bump counter, return the cached answer (stored in metadata)
            meta = self._collection.get(ids=[results["ids"][0][0]], include=["metadatas"])
            full_meta = meta["metadatas"][0] if meta["metadatas"] else {}
            answer = full_meta.get("answer", "")
            count = full_meta.get("count", 1)
            rating = full_meta.get("rating", 0)
            self._collection.update(
                ids=[results["ids"][0][0]],
                metadatas=[{"answer": answer, "count": count + 1, "rating": rating}],
            )
            if self.adaptive:
                self._adjust()
            self._save_stats()
            return answer, similarity

        if self.adaptive:
            self._adjust()
        self._save_stats()
        return None, similarity

    def store(self, question: str, answer: str) -> None:
        """Store a new question-answer pair."""
        import uuid

        embedding = self._model.encode([question])[0].tolist()
        self._collection.add(
            embeddings=[embedding],
            documents=[question],
            metadatas=[{"answer": answer, "count": 1, "rating": 0}],
            ids=[str(uuid.uuid4())],
        )

    def feedback(self, question: str, positive: bool) -> None:
        """Record 👍 (positive=True) or 👎 feedback on the last match for this question."""
        embedding = self._model.encode([question])[0].tolist()
        results = self._collection.query(
            query_embeddings=[embedding], n_results=1, include=["metadatas"]
        )
        if not results["ids"][0]:
            return

        meta = results["metadatas"][0][0]
        delta = 1 if positive else -1
        new_rating = meta.get("rating", 0) + delta

        # Remove FAQ if too many downvotes
        if new_rating <= -3:
            self._collection.delete(ids=[results["ids"][0][0]])
        else:
            self._collection.update(
                ids=[results["ids"][0][0]],
                metadatas=[{
                    "answer": meta.get("answer", ""),
                    "count": meta.get("count", 1),
                    "rating": new_rating,
                }],
            )

    @property
    def stats(self) -> dict:
        return {
            "total_queries": self._total,
            "cache_hits": self._hits,
            "hit_rate": round(self._hits / max(self._total, 1), 3),
            "cached_faqs": self._collection.count(),
            "threshold": round(self.threshold, 3),
            "adaptive": self.adaptive,
        }

    # ── internal ────────────────────────────────────────────

    def _adjust(self) -> None:
        if self._total == 0:
            return
        hit_rate = self._hits / self._total
        if hit_rate < self.target_hit_rate:
            self.threshold = max(0.55, self.threshold - self.adjustment_rate)
        else:
            self.threshold = min(0.98, self.threshold + self.adjustment_rate)

    def _load_stats(self) -> None:
        if self._stats_path.exists():
            data = json.loads(self._stats_path.read_text())
            self._total = data.get("total", 0)
            self._hits = data.get("hits", 0)

    def _save_stats(self) -> None:
        self._stats_path.write_text(json.dumps({
            "total": self._total, "hits": self._hits
        }))
