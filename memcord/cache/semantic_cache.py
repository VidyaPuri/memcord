"""Semantic answer cache — ChromaDB + pluggable embeddings + scope + hooks + promotion.

This is the reusable core.  FAQCache is a thin backward-compatible subclass.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import chromadb

from dataclasses import dataclass


@dataclass
class CacheHit:
    """A cache hit — returned by lookup()."""

    id: str
    answer: str
    similarity: float = 0.0

log = logging.getLogger("memcord.cache")

# Schema version for migration support
SCHEMA_VERSION = 2


class SemanticAnswerCache:
    """Semantically caches question-answer pairs with scope partitioning.

    Public surface::

        cache = SemanticAnswerCache(data_dir="./data")
        hit  = cache.lookup("how do I install?", scope="support")
        if hit is None:
            answer = llm.ask("how do I install?")
            cache.observe("how do I install?", answer, scope="support")
        else:
            print(hit.answer)

    Scope is an opaque partition key — entries stored under one scope are
    never returned for another.  ``should_cache`` and ``validate`` are
    caller-supplied callbacks that gate eligibility and retrieval.
    ``promote_after`` controls how many times a question must be observed
    before it becomes retrievable via ``lookup``.
    """

    def __init__(
        self,
        data_dir: str = "./memcord_data",
        *,
        scope_default: str = "default",
        promote_after: int = 1,
        should_cache: Callable[[str], bool] | None = None,
        validate: Callable[[CacheHit], bool] | None = None,
        embed_model: object | None = None,
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

        # ── hooks ────────────────────────────────────
        self._scope_default = scope_default
        self._promote_after = max(1, promote_after)
        self._should_cache = should_cache if should_cache is not None else (lambda _: True)
        self._validate = validate if validate is not None else (lambda _: True)
        self._embed_model_spec = embed_model

        # Embedding model — lazy loaded on first use
        self.__model: object | None = None

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

        # Feedback threshold
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

    # ── public API ──────────────────────────────────────────

    def lookup(self, question: str, *, scope: str = "") -> CacheHit | None:
        """Check if a similar question is cached *in the given scope*.

        Returns a ``CacheHit`` on success, ``None`` on miss.
        Candidates that haven't reached ``promote_after`` observations or
        that fail the ``validate`` hook are discarded.
        """
        scope = scope or self._scope_default
        self._total += 1

        if self._collection.count() == 0:
            self._maybe_save_stats()
            return None

        embedding = self._model.encode([question])[0].tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=10,  # fetch more so we can filter in Python
            where={"scope": scope},
            include=["documents", "metadatas", "distances"],
        )

        if not results["ids"][0]:
            self._maybe_save_stats()
            return None

        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i]
            similarity = 1.0 - distance

            if similarity < self.threshold:
                continue

            hit_id = results["ids"][0][i]
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            answer = meta.get("answer", "")
            obs_count = meta.get("observation_count", 1)

            # promote_after gate — entry invisible until observed enough
            if obs_count < self._promote_after:
                continue

            # validate hook
            candidate = CacheHit(id=hit_id, answer=answer)
            if not self._validate(candidate):
                continue

            # Hit! Update usage stats
            self._hits += 1
            count = meta.get("count", 1)
            rating = meta.get("rating", 0)
            created_at = meta.get("created_at", 0)
            self._collection.update(
                ids=[hit_id],
                metadatas=[
                    {
                        "answer": answer,
                        "count": count + 1,
                        "rating": rating,
                        "created_at": created_at,
                        "scope": scope,
                        "observation_count": obs_count,
                    }
                ],
            )
            if self.adaptive:
                self._adjust()
            self._maybe_save_stats()
            return CacheHit(id=hit_id, answer=answer, similarity=similarity)

        if self.adaptive:
            self._adjust()
        self._maybe_save_stats()
        return None

    def observe(self, question: str, answer: str, *, scope: str = "") -> None:
        """Record a question→answer pair.

        If ``should_cache(answer)`` returns ``False`` the pair is silently
        dropped.  Otherwise the pair is stored and its ``observation_count``
        incremented.  A question becomes retrievable via ``lookup`` only
        after it has been observed ``promote_after`` times.
        """
        scope = scope or self._scope_default

        # should_cache gate
        if not self._should_cache(answer):
            return

        self._stores += 1
        embedding = self._model.encode([question])[0].tolist()

        # Check for near-duplicate within the same scope
        if self._collection.count() > 0:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=1,
                where={"scope": scope},
                include=["metadatas", "distances"],
            )
            if results["ids"][0]:
                similarity = 1.0 - results["distances"][0][0]
                if similarity >= self.consolidate_threshold:
                    meta = results["metadatas"][0][0] if results["metadatas"] else {}
                    existing_rating = meta.get("rating", 0)
                    existing_count = meta.get("count", 1)
                    existing_answer = meta.get("answer", "")
                    obs_count = meta.get("observation_count", 1) + 1
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
                                "scope": scope,
                                "observation_count": obs_count,
                            }
                        ],
                    )
                    self._maybe_save_stats()
                    return

        # New entry
        self._collection.add(
            embeddings=[embedding],
            documents=[question],
            metadatas=[
                {
                    "answer": answer,
                    "count": 1,
                    "rating": 0,
                    "created_at": time.time(),
                    "scope": scope,
                    "observation_count": 1,
                }
            ],
            ids=[str(uuid.uuid4())],
        )
        self._maybe_save_stats()

    def vote(self, hit_id: str, delta: int) -> None:
        """Adjust the quality rating of a cached entry.

        ``delta`` is typically +1 (upvote) or -1 (downvote).
        Entries whose rating falls to ≤ -3 are automatically pruned.
        If ``hit_id`` is not found this is a no-op.
        """
        try:
            results = self._collection.get(
                ids=[hit_id],
                include=["metadatas"],
            )
        except Exception:
            return  # ChromaDB error on missing ID — treat as no-op

        if not results["ids"]:
            return

        if delta > 0:
            self._upvotes += 1
        else:
            self._downvotes += 1

        meta = results["metadatas"][0] if results["metadatas"] else {}
        new_rating = meta.get("rating", 0) + delta

        if new_rating <= -3:
            self._collection.delete(ids=[hit_id])
            self._deletions += 1
            if self._metrics and hasattr(self._metrics, "faq_deletion"):
                self._metrics.faq_deletion()
        else:
            self._collection.update(
                ids=[hit_id],
                metadatas=[
                    {
                        "answer": meta.get("answer", ""),
                        "count": meta.get("count", 1),
                        "rating": new_rating,
                        "created_at": meta.get("created_at", 0),
                        "scope": meta.get("scope", self._scope_default),
                        "observation_count": meta.get("observation_count", 1),
                    }
                ],
            )

        self._maybe_save_stats()

    # ── utility methods ─────────────────────────────────────

    def set_metrics(self, metrics: object) -> None:
        """Wire an external Metrics instance for deletion tracking."""
        self._metrics = metrics

    def consolidate(self) -> int:
        """Scan all FAQs and merge near-duplicates (any scope).

        Returns the number of FAQs merged.
        """
        count = self._collection.count()
        if count < 2:
            return 0

        all_data = self._collection.get(include=["embeddings", "metadatas", "documents"])
        if not all_data["ids"]:
            return 0

        ids = all_data["ids"]
        embeddings = all_data["embeddings"]
        metadatas = all_data["metadatas"]

        merged = 0
        for i in range(len(ids)):
            if ids[i] is None:
                continue
            emb_i = embeddings[i]
            for j in range(i + 1, len(ids)):
                if ids[j] is None:
                    continue
                emb_j = embeddings[j]
                dot = sum(a * b for a, b in zip(emb_i, emb_j, strict=False))
                if dot >= self.consolidate_threshold:
                    meta_i = metadatas[i] or {}
                    meta_j = metadatas[j] or {}
                    # Only merge if same scope
                    if meta_i.get("scope") != meta_j.get("scope"):
                        continue
                    rating_i = meta_i.get("rating", 0)
                    rating_j = meta_j.get("rating", 0)
                    count_i = meta_i.get("count", 1)
                    count_j = meta_j.get("count", 1)
                    obs_i = meta_i.get("observation_count", 1)
                    obs_j = meta_j.get("observation_count", 1)

                    if rating_j > rating_i:
                        winner_id, loser_id = ids[j], ids[i]
                        winner_meta = meta_j
                        winner_count, loser_count = count_j, count_i
                        winner_obs, loser_obs = obs_j, obs_i
                    else:
                        winner_id, loser_id = ids[i], ids[j]
                        winner_meta = meta_i
                        winner_count, loser_count = count_i, count_j
                        winner_obs, loser_obs = obs_i, obs_j

                    self._collection.update(
                        ids=[winner_id],
                        metadatas=[
                            {
                                "answer": winner_meta.get("answer", ""),
                                "count": winner_count + loser_count,
                                "rating": winner_meta.get("rating", 0),
                                "created_at": winner_meta.get("created_at", 0),
                                "scope": winner_meta.get("scope", self._scope_default),
                                "observation_count": max(winner_obs, loser_obs),
                            }
                        ],
                    )
                    self._collection.delete(ids=[loser_id])
                    if self._metrics and hasattr(self._metrics, "faq_deletion"):
                        self._metrics.faq_deletion()
                    merged += 1
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
        scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """List or search FAQs with pagination.

        Args:
            query: Optional search query for semantic similarity search.
            limit: Max results to return (1-100).
            offset: Number of results to skip.
            include_embeddings: If True, include embedding vectors.
            scope: Optional scope filter.

        Returns:
            List of dicts with keys: id, question, answer, count, rating,
            similarity (if query), observation_count.
        """
        limit = max(1, min(limit, 100))

        if query:
            embedding = self._model.encode([query])[0].tolist()
            kwargs: dict[str, Any] = {
                "query_embeddings": [embedding],
                "n_results": min(limit + offset, self._collection.count()),
                "include": ["documents", "metadatas", "distances"]
                + (["embeddings"] if include_embeddings else []),
            }
            if scope is not None:
                kwargs["where"] = {"scope": scope}
            results = self._collection.query(**kwargs)
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
                    "observation_count": meta.get("observation_count", 1),
                    "scope": meta.get("scope", self._scope_default),
                    "similarity": round(similarity, 4),
                }
                if include_embeddings:
                    entry["embedding"] = results.get("embeddings", [[]])[0][i]
                faqs.append(entry)
                if len(faqs) >= limit:
                    break
        else:
            kwargs = {
                "limit": limit,
                "offset": offset,
                "include": ["documents", "metadatas"]
                + (["embeddings"] if include_embeddings else []),
            }
            if scope is not None:
                kwargs["where"] = {"scope": scope}
            results = self._collection.get(**kwargs)
            faqs = []
            for i in range(len(results["ids"])):
                meta = results["metadatas"][i] or {}
                entry = {
                    "id": results["ids"][i],
                    "question": results["documents"][i],
                    "answer": meta.get("answer", ""),
                    "count": meta.get("count", 1),
                    "rating": meta.get("rating", 0),
                    "created_at": meta.get("created_at", 0),
                    "observation_count": meta.get("observation_count", 1),
                    "scope": meta.get("scope", self._scope_default),
                }
                if include_embeddings:
                    entry["embedding"] = results.get("embeddings", [])[i]
                faqs.append(entry)

        return faqs

    def reset(self) -> None:
        """Reset the cache completely (for debugging/testing)."""
        self._chroma.delete_collection("faqs")
        self._collection = self._chroma.get_or_create_collection(
            name="faqs",
            metadata={"hnsw:space": "cosine", "schema_version": SCHEMA_VERSION},
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

    def flush_stats(self) -> None:
        """Force-write stats to disk immediately."""
        self._save_stats()

    # ── lazy model loading ──────────────────────────────────

    @property
    def _model(self):
        """Lazy-load the embedding model on first access."""
        if self.__model is None:
            from memcord.embedding import resolve_embed_model

            self.__model = resolve_embed_model(self._embed_model_spec)
        return self.__model

    # ── migration ───────────────────────────────────────────

    def _get_or_migrate_collection(self):
        """Get or create collection with migration support."""
        collection = self._chroma.get_or_create_collection(
            name="faqs", metadata={"hnsw:space": "cosine"}
        )

        existing_meta = collection.metadata or {}
        stored_version = existing_meta.get("schema_version", 0)

        if stored_version < SCHEMA_VERSION:
            self._migrate(collection, stored_version)
            new_meta = dict(existing_meta)
            new_meta["schema_version"] = SCHEMA_VERSION
            new_meta.pop("hnsw:space", None)
            collection.modify(metadata=new_meta)

        return collection

    def _migrate(self, collection, from_version: int) -> None:
        """Apply schema migrations from from_version to SCHEMA_VERSION."""
        # Migration 0 → 1: no structural changes (legacy)
        if from_version < 1:
            pass

        # Migration 1 → 2: add scope and observation_count to existing entries
        if from_version < 2:
            count = collection.count()
            if count == 0:
                return
            # Process in batches
            batch_size = 100
            for offset in range(0, count, batch_size):
                data = collection.get(
                    limit=batch_size,
                    offset=offset,
                    include=["metadatas"],
                )
                ids_to_update = []
                metas_to_update = []
                for i, meta in enumerate(data["metadatas"]):
                    if meta is None:
                        meta = {}
                    if "scope" not in meta or "observation_count" not in meta:
                        ids_to_update.append(data["ids"][i])
                        new_meta = dict(meta)
                        new_meta.setdefault("scope", self._scope_default)
                        # Existing entries are considered well-established
                        new_meta.setdefault("observation_count", max(meta.get("count", 1), 100))
                        metas_to_update.append(new_meta)
                if ids_to_update:
                    collection.update(ids=ids_to_update, metadatas=metas_to_update)
                    log.info(
                        "Migrated %d entries to schema v2 (scope + observation_count)",
                        len(ids_to_update),
                    )

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


# ── factory ─────────────────────────────────────────────────


def build_cache(
    *,
    scope_default: str = "default",
    promote_after: int = 1,
    should_cache: Callable[[str], bool] | None = None,
    validate: Callable[[CacheHit], bool] | None = None,
    **provider_options,
) -> SemanticAnswerCache:
    """Build a SemanticAnswerCache instance.

    All keyword arguments are forwarded to ``SemanticAnswerCache.__init__``.

    Required provider option:
        ``data_dir`` (str) — path to the cache data directory.

    Common optional provider options:
        ``embed_model``, ``similarity_threshold``, ``adaptive_threshold``,
        ``consolidate_threshold``.

    Args:
        scope_default: Default scope when ``lookup``/``observe`` is called without one.
        promote_after: Entries become retrievable only after N observations (default 1).
        should_cache: Optional callable(answer) → bool; False drops the answer.
        validate: Optional callable(CacheHit) → bool; False discards the candidate.
        **provider_options: Passed through to ``SemanticAnswerCache.__init__``.
          Must include ``data_dir``.
    """
    return SemanticAnswerCache(
        scope_default=scope_default,
        promote_after=promote_after,
        should_cache=should_cache,
        validate=validate,
        **provider_options,
    )
