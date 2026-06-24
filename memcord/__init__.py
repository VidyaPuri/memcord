"""Memcord — LLM-agnostic Discord FAQ bot with semantic caching."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("memcord")
except PackageNotFoundError:
    __version__ = "0.1.0+local"
