"""LLM backends — pluggable, ABC-driven, with retry/timeout/circuit-breaker."""

from __future__ import annotations

import importlib
import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger("memcord.backends")


class LLMBackend(ABC):
    """Abstract base for every LLM backend."""

    @abstractmethod
    async def ask(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt, return the response text."""
        ...


# ── factory ────────────────────────────────────────────────────────────


def get_backend() -> LLMBackend:
    """Instantiate the backend configured by environment variables.

    Controlled by:
    * ``MEMCORD_BACKEND`` – one of claude_code, openai, anthropic, ollama, custom
    * ``MEMCORD_MODEL`` – override the per-backend default model
    * ``MEMCORD_CUSTOM_BACKEND`` – for ``custom``, dotted path e.g. "mypkg.MyBackend"
    * ``MEMCORD_TIMEOUT`` / ``MEMCORD_RETRY_MAX`` / ``MEMCORD_RETRY_DELAY`` –
      passed through to the base class (see ``BaseBackend``)
    """
    backend_name = os.getenv("MEMCORD_BACKEND", "claude_code")
    model = os.getenv("MEMCORD_MODEL", "")

    if backend_name == "claude_code":
        from memcord.backends.claude_code import ClaudeCodeBackend

        return ClaudeCodeBackend(model=model or "sonnet")

    if backend_name == "openai":
        from memcord.backends.openai_backend import OpenAIBackend

        return OpenAIBackend(model=model or "gpt-4o")

    if backend_name == "anthropic":
        from memcord.backends.anthropic_backend import AnthropicBackend

        return AnthropicBackend(model=model or "claude-sonnet-4-20250514")

    if backend_name == "ollama":
        from memcord.backends.ollama_backend import OllamaBackend

        return OllamaBackend(model=model or "llama3")

    if backend_name == "custom":
        path = os.getenv("MEMCORD_CUSTOM_BACKEND", "")
        if not path:
            raise ValueError("MEMCORD_CUSTOM_BACKEND must be set (e.g. 'mypackage.MyBackend')")
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)()

    raise ValueError(f"Unknown backend: {backend_name}")
