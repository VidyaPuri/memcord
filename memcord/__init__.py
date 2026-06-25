"""Memcord — LLM-agnostic Discord FAQ bot with semantic caching."""

from importlib.metadata import PackageNotFoundError, version

from memcord.cache import AnswerCache, CacheHit, FAQCache, SemanticAnswerCache, build_cache

try:
    __version__ = version("memcord")
except PackageNotFoundError:
    __version__ = "0.1.0+local"

__all__ = [
    "__version__",
    "FAQCache",
    "SemanticAnswerCache",
    "build_cache",
    "CacheHit",
    "AnswerCache",
]
