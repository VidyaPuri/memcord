"""Semantic FAQ cache — public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memcord.cache.faq_cache import FAQCache


@dataclass
class CacheHit:
    """A cache hit — returned by lookup()."""

    id: str
    answer: str


class AnswerCache(Protocol):
    """Protocol for any cache implementing the semantic-cache interface."""

    def lookup(self, question: str, *, scope: str) -> CacheHit | None: ...
    def observe(self, question: str, answer: str, *, scope: str) -> None: ...
    def vote(self, hit_id: str, delta: int) -> None: ...


__all__ = [
    "FAQCache",
    "CacheHit",
    "AnswerCache",
]
