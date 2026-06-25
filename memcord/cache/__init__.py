"""Semantic FAQ cache — public API."""

from __future__ import annotations

from typing import Protocol

from memcord.cache.faq_cache import FAQCache
from memcord.cache.semantic_cache import CacheHit, SemanticAnswerCache, build_cache


class AnswerCache(Protocol):
    """Protocol for any cache implementing the semantic-cache interface."""

    def lookup(self, question: str, *, scope: str) -> CacheHit | None: ...
    def observe(self, question: str, answer: str, *, scope: str) -> None: ...
    def vote(self, hit_id: str, delta: int) -> None: ...


__all__ = [
    "FAQCache",
    "SemanticAnswerCache",
    "build_cache",
    "CacheHit",
    "AnswerCache",
]
