"""Ollama backend — local LLM via Ollama REST API."""

from __future__ import annotations

import logging
import os

from memcord.backends.base import BaseBackend, RetryableError

logger = logging.getLogger("memcord.backends.ollama")

try:
    import httpx

    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False
    httpx = None  # type: ignore[assignment,misc]


class OllamaBackend(BaseBackend):
    """Backend for Ollama's local REST API.

    Handles:
    * ``httpx.ConnectError``      → retryable
    * ``httpx.ReadTimeout``       → retryable
    * ``httpx.HTTPStatusError``   → retryable for 5xx, fatal for 4xx
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        if not _HTTPX_AVAILABLE:
            raise ImportError("pip install memcord[ollama] to use Ollama backend")
        super().__init__(
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.model = model
        self.base_url = base_url or os.getenv(
            "MEMCORD_OLLAMA_URL", "http://localhost:11434"
        )

    async def _ask_impl(self, prompt: str, system: str | None = None) -> str:
        payload: dict = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/generate", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["response"]

    # ── classify subclass-specific errors ──────────────────────────────

    def _is_retryable(self, error: Exception) -> bool:
        if isinstance(error, RetryableError):
            return True

        if _HTTPX_AVAILABLE:
            # Connection errors → retry (Ollama might be starting up)
            if isinstance(error, httpx.ConnectError):
                return True
            if isinstance(error, httpx.ReadTimeout):
                return True

            # HTTP status errors: retry on 5xx, fail on 4xx
            if isinstance(error, httpx.HTTPStatusError):
                code = error.response.status_code
                if 500 <= code < 600:
                    return True
                if 400 <= code < 500:
                    return False

        return super()._is_retryable(error)
