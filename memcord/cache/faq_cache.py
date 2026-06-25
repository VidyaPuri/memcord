"""FAQCache — backward-compatible wrapper around SemanticAnswerCache.

This is the class the Discord bot imports.  It exposes the original
``check`` / ``store`` / ``feedback`` API on top of SemanticAnswerCache.
"""

from __future__ import annotations

import logging

from memcord.cache.semantic_cache import CacheHit, SemanticAnswerCache

log = logging.getLogger("memcord.cache")


class FAQCache(SemanticAnswerCache):
    """Backward-compatible FAQ cache.

    Wraps SemanticAnswerCache with the original check/store/feedback API.
    All queries use ``scope_default`` (typically "default").
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
        super().__init__(
            data_dir=data_dir,
            scope_default="default",
            promote_after=1,
            should_cache=None,
            validate=None,
            embed_model=model_name,
            similarity_threshold=similarity_threshold,
            adaptive_threshold=adaptive_threshold,
            target_hit_rate=target_hit_rate,
            adjustment_rate=adjustment_rate,
            feedback_threshold=feedback_threshold,
            stats_batch_size=stats_batch_size,
            stats_batch_seconds=stats_batch_seconds,
            consolidate_threshold=consolidate_threshold,
        )

    # ── legacy API ──────────────────────────────────────────

    def check(self, question: str) -> tuple[str | None, float]:
        """Check if a similar question is cached.

        Returns (answer, similarity) or (None, 0.0) on miss.
        """
        hit = self.lookup(question)
        if hit is None:
            return None, 0.0
        return hit.answer, hit.similarity

    def store(self, question: str, answer: str) -> None:
        """Store a new question-answer pair."""
        self.observe(question, answer)

    def feedback(self, question: str, positive: bool) -> None:
        """Record 👍 or 👎 feedback on the last match for this question."""
        embedding = self._model.encode([question])[0].tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=1,
            where={"scope": self._scope_default},
            include=["metadatas", "distances"],
        )
        if not results["ids"][0]:
            return

        # Verify this is actually the right FAQ (similarity check)
        similarity = 1.0 - results["distances"][0][0]
        if similarity < self.feedback_threshold:
            return  # wrong FAQ matched — skip feedback

        delta = 1 if positive else -1
        self.vote(results["ids"][0][0], delta)
