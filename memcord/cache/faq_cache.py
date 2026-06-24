"""Semantic FAQ cache — ChromaDB + SentenceBERT + adaptive threshold."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

import chromadb

log = logging.getLogger("memcord.cache")


# Schema version for migration support
SCHEMA_VERSION = 1


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
        feedback_threshold: float | None = None,
        stats_batch_size: int = 10,
        stats_batch_seconds: float = 60.0,
        consolidate_threshold: float = 0.95,
    ):
        os.makedirs(data_dir, exist_ok=True)

        # Embedding model — lazy loaded on first use
        self._model_name = model_name
        self.__model = None  # type: ignore[assignment]

        # ChromaDB
        self._chroma = chromadb.PersistentClient(
            path=str(Path(data_dir) / "chroma"),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self._collection = self._get_or_migrate_collection()

        # Threshold
        self.threshold = similarity_threshold
        self.adaptive = adaptive_threshold
        self.target_hit_rate = target_hit_rate
        self.adjustment_rate = adjustment_rate

        # Feedback threshold — env var MEMCORD_FEEDBACK_THRESHOLD or default 0.80
        if feedback_threshold is None:
            feedback_threshold = float(os.getenv("MEMCORD_FEEDBACK_THRESHOLD", "0.80"))
        self.feedback_threshold = feedback_threshold

        # Consolidation threshold
        self.consolidate_threshold = consolidate_threshold

        # Stats
        self._total = 0
        self._hits = 0
        self._upvotes = 0
        self._downvotes = 0
        self._deletions = 0
        self._stores = 0
        self._stats_path = Path(data_dir) / "stats.json"
        self._load_stats()

        # Batched stats
        self._stats_queries_since_save = 0
        self._last_stats_save = time.time()
        self._stats_batch_size = stats_batch_size
        self._stats_batch_seconds = stats_batch_seconds

        # Metrics hook (optional)
        self._metrics: object | None = None

    def set_metrics(self, metrics: object) -> None:
        """Wire an external Metrics instance for deletion tracking."""
        self._metrics = metrics

    # ── lazy model loading ──────────────────────────────────

    @property
    def _model(self):
        """Lazy-load the SentenceTransformer model on first access."""
        if self.__model is None:
            from sentence_transformers import SentenceTransformer

            self.__model = SentenceTransformer(self._model_name)
        return self.__model

    # ── migration ───────────────────────────────────────────

    def _get_or_migrate_collection(self):
        """Get or create collection with migration support."""
        collection = self._chroma.get_or_create_collection(
            name="faqs", metadata={"hnsw:space": "cosine"}
        )

        # Check schema version stored in collection metadata
        existing_meta = collection.metadata or {}
        stored_version = existing_meta.get("schema_version", 0)

        if stored_version < SCHEMA_VERSION:
            # Run migrations
            self._migrate(collection, stored_version)
            # Update schema version (preserve hnsw:space and other existing keys)
            new_meta = dict(existing_meta)
            new_meta["schema_version"] = SCHEMA_VERSION
            # Remove hnsw:space from modify call — it can't be changed after creation
            new_meta.pop("hnsw:space", None)
            collection.modify(metadata=new_meta)

        return collection

    def _migrate(self, collection, from_version: int) -> None:
        """Apply schema migrations from from_version to SCHEMA_VERSION."""
        # Migration 0 → 1: ensure metadata fields exist (count, rating, answer)
        if from_version < 1:
            # No structural changes needed — metadata fields are validated at access time
            pass

    # ── public API ──────────────────────────────────────────

    def check(self, question: str) -> tuple[str | None, float]:
        """Check if a similar question is cached.

        Returns (answer, similarity_score) or (None, 0.0) on miss.
        """
        self._total += 1

        if self._collection.count() == 0:
            self._maybe_save_stats()
            return None, 0.0

        embedding = self._model.encode([question])[0].tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            self._maybe_save_stats()
            return None, 0.0

        distance = results["distances"][0][0]
        similarity = 1.0 - distance  # cosine → similarity

        if similarity >= self.threshold:
            self._hits += 1
            # Read answer directly from query result metadata (no second DB call)
            full_meta = results["metadatas"][0][0] if results["metadatas"] else {}
            answer = full_meta.get("answer", "")
            count = full_meta.get("count", 1)
            rating = full_meta.get("rating", 0)
            created_at = full_meta.get("created_at", 0)
            self._collection.update(
                ids=[results["ids"][0][0]],
                metadatas=[
                    {
                        "answer": answer,
                        "count": count + 1,
                        "rating": rating,
                        "created_at": created_at,
                    }
                ],
            )
            if self.adaptive:
                self._adjust()
            self._maybe_save_stats()
            return answer, similarity

        if self.adaptive:
            self._adjust()
        self._maybe_save_stats()
        return None, similarity

    def store(self, question: str, answer: str) -> None:
        """Store a new question-answer pair, with optional duplicate consolidation."""
        self._stores += 1

        embedding = self._model.encode([question])[0].tolist()

        # Check for near-duplicate before storing
        if self._collection.count() > 0:
            results = self._collection.query(
                query_embeddings=[embedding], n_results=1, include=["metadatas", "distances"]
            )
            if results["ids"][0]:
                similarity = 1.0 - results["distances"][0][0]
                if similarity >= self.consolidate_threshold:
                    # Near-duplicate found — update existing instead of adding new
                    meta = results["metadatas"][0][0] if results["metadatas"] else {}
                    existing_rating = meta.get("rating", 0)
                    existing_count = meta.get("count", 1)
                    existing_answer = meta.get("answer", "")

                    # Keep the answer with the higher rating, or the new one if tied
                    keep_answer = answer if existing_rating <= 0 else existing_answer
                    existing_created_at = meta.get("created_at", 0)

                    self._collection.update(
                        ids=[results["ids"][0][0]],
                        metadatas=[
                            {
                                "answer": keep_answer,
                                "count": existing_count + 1,
                                "rating": existing_rating,
                                "created_at": existing_created_at,
                            }
                        ],
                    )
                    self._maybe_save_stats()
                    return

        self._collection.add(
            embeddings=[embedding],
            documents=[question],
            metadatas=[{"answer": answer, "count": 1, "rating": 0, "created_at": time.time()}],
            ids=[str(uuid.uuid4())],
        )
        self._maybe_save_stats()

    def feedback(self, question: str, positive: bool) -> None:
        """Record 👍 (positive=True) or 👎 feedback on the last match for this question."""
        embedding = self._model.encode([question])[0].tolist()
        results = self._collection.query(
            query_embeddings=[embedding], n_results=1, include=["metadatas", "distances"]
        )
        if not results["ids"][0]:
            return

        # Verify this is actually the right FAQ (similarity check)
        similarity = 1.0 - results["distances"][0][0]
        if similarity < self.feedback_threshold:
            return  # wrong FAQ matched — skip feedback

        if positive:
            self._upvotes += 1
        else:
            self._downvotes += 1

        meta = results["metadatas"][0][0]
        delta = 1 if positive else -1
        new_rating = meta.get("rating", 0) + delta

        # Remove FAQ if too many downvotes
        if new_rating <= -3:
            self._collection.delete(ids=[results["ids"][0][0]])
            self._deletions += 1
            if self._metrics and hasattr(self._metrics, "faq_deletion"):
                self._metrics.faq_deletion()
        else:
            self._collection.update(
                ids=[results["ids"][0][0]],
                metadatas=[
                    {
                        "answer": meta.get("answer", ""),
                        "count": meta.get("count", 1),
                        "rating": new_rating,
                        "created_at": meta.get("created_at", 0),
                    }
                ],
            )

        self._maybe_save_stats()

    def consolidate(self) -> int:
        """Scan all FAQs and merge near-duplicates (similarity > consolidate_threshold).

        Returns the number of FAQs merged.
        """
        count = self._collection.count()
        if count < 2:
            return 0

        # Get all FAQs with embeddings
        all_data = self._collection.get(include=["embeddings", "metadatas", "documents"])
        if not all_data["ids"]:
            return 0

        ids = all_data["ids"]
        embeddings = all_data["embeddings"]
        metadatas = all_data["metadatas"]

        merged = 0
        # Compare each pair — O(n²) but expected FAQ count is low
        for i in range(len(ids)):
            if ids[i] is None:
                continue
            emb_i = embeddings[i]
            for j in range(i + 1, len(ids)):
                if ids[j] is None:
                    continue
                emb_j = embeddings[j]
                # Cosine similarity via dot product (normalized embeddings)
                dot = sum(a * b for a, b in zip(emb_i, emb_j, strict=False))
                if dot >= self.consolidate_threshold:
                    # Merge j into i: keep the better-rated answer
                    meta_i = metadatas[i] or {}
                    meta_j = metadatas[j] or {}
                    rating_i = meta_i.get("rating", 0)
                    rating_j = meta_j.get("rating", 0)
                    count_i = meta_i.get("count", 1)
                    count_j = meta_j.get("count", 1)

                    if rating_j > rating_i:
                        # j has better answer, merge i into j
                        winner_id, loser_id = ids[j], ids[i]
                        winner_meta = meta_j
                        winner_count, loser_count = count_j, count_i
                    else:
                        # i keeps its answer (or tie)
                        winner_id, loser_id = ids[i], ids[j]
                        winner_meta = meta_i
                        winner_count, loser_count = count_i, count_j

                    # Update winner with combined counts
                    self._collection.update(
                        ids=[winner_id],
                        metadatas=[
                            {
                                "answer": winner_meta.get("answer", ""),
                                "count": winner_count + loser_count,
                                "rating": winner_meta.get("rating", 0),
                                "created_at": winner_meta.get("created_at", 0),
                            }
                        ],
                    )
                    # Delete loser
                    self._collection.delete(ids=[loser_id])
                    if self._metrics and hasattr(self._metrics, "faq_deletion"):
                        self._metrics.faq_deletion()
                    merged += 1
                    # Mark loser as None so we don't process it again
                    ids[j] = None

        if merged > 0:
            self._deletions += merged
            self._maybe_save_stats()
        return merged

    def list_faqs(
        self,
        query: str | None = None,
        limit: int = 10,
        offset: int = 0,
        include_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        """List or search FAQs with pagination.

        Args:
            query: Optional search query for semantic similarity search.
            limit: Max results to return (1-100).
            offset: Number of results to skip.
            include_embeddings: If True, include embedding vectors.

        Returns:
            List of dicts with keys: id, question, answer, count, rating, similarity (if query).
        """
        limit = max(1, min(limit, 100))

        if query:
            embedding = self._model.encode([query])[0].tolist()
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(limit + offset, self._collection.count()),
                include=["documents", "metadatas", "distances"]
                + (["embeddings"] if include_embeddings else []),
            )
            faqs = []
            for i in range(min(len(results["ids"][0]), limit + offset)):
                if i < offset:
                    continue
                similarity = 1.0 - results["distances"][0][i]
                meta = results["metadatas"][0][i] or {}
                entry: dict[str, Any] = {
                    "id": results["ids"][0][i],
                    "question": results["documents"][0][i],
                    "answer": meta.get("answer", ""),
                    "count": meta.get("count", 1),
                    "rating": meta.get("rating", 0),
                    "created_at": meta.get("created_at", 0),
                    "similarity": round(similarity, 4),
                }
                if include_embeddings:
                    entry["embedding"] = results.get("embeddings", [[]])[0][i]
                faqs.append(entry)
                if len(faqs) >= limit:
                    break
        else:
            results = self._collection.get(
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"] + (["embeddings"] if include_embeddings else []),
            )
            faqs = []
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i] or {}
                entry: dict[str, Any] = {
                    "id": results["ids"][i],
                    "question": results["documents"][i],
                    "answer": meta.get("answer", ""),
                    "count": meta.get("count", 1),
                    "rating": meta.get("rating", 0),
                    "created_at": meta.get("created_at", 0),
                }
                if include_embeddings:
                    entry["embedding"] = results.get("embeddings", [])[i]
                faqs.append(entry)

        return faqs

    def reset(self) -> None:
        """Reset the cache completely (for debugging/testing)."""
        self._chroma.delete_collection("faqs")
        self._collection = self._chroma.get_or_create_collection(
            name="faqs", metadata={"hnsw:space": "cosine", "schema_version": SCHEMA_VERSION}
        )
        self._total = 0
        self._hits = 0
        self._upvotes = 0
        self._downvotes = 0
        self._deletions = 0
        self._stores = 0
        self._stats_queries_since_save = 0
        self._last_stats_save = time.time()
        self._save_stats()

    @property
    def stats(self) -> dict:
        return {
            "total_queries": self._total,
            "cache_hits": self._hits,
            "hit_rate": round(self._hits / max(self._total, 1), 3),
            "cached_faqs": self._collection.count(),
            "threshold": round(self.threshold, 3),
            "adaptive": self.adaptive,
            "upvotes": self._upvotes,
            "downvotes": self._downvotes,
            "deletions": self._deletions,
            "stores": self._stores,
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
            self._upvotes = data.get("upvotes", 0)
            self._downvotes = data.get("downvotes", 0)
            self._deletions = data.get("deletions", 0)
            self._stores = data.get("stores", 0)

    def _save_stats(self) -> None:
        self._stats_path.write_text(
            json.dumps(
                {
                    "total": self._total,
                    "hits": self._hits,
                    "upvotes": self._upvotes,
                    "downvotes": self._downvotes,
                    "deletions": self._deletions,
                    "stores": self._stores,
                }
            )
        )
        self._stats_queries_since_save = 0
        self._last_stats_save = time.time()

    def _maybe_save_stats(self) -> None:
        """Save stats only if the batch threshold is reached."""
        self._stats_queries_since_save += 1
        now = time.time()
        if (
            self._stats_queries_since_save >= self._stats_batch_size
            or (now - self._last_stats_save) >= self._stats_batch_seconds
        ):
            self._save_stats()

    def flush_stats(self) -> None:
        """Force-write stats to disk immediately."""
        self._save_stats()
