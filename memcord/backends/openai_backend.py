"""OpenAI API backend."""

from __future__ import annotations

import logging

from memcord.backends.base import BaseBackend, RetryableError

logger = logging.getLogger("memcord.backends.openai")

try:
    import openai
    from openai import AsyncOpenAI

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore[assignment,misc]


class OpenAIBackend(BaseBackend):
    """Backend using the OpenAI Python SDK.

    Handles:
    * ``RateLimitError``         → retryable
    * ``APITimeoutError``        → retryable
    * ``APIConnectionError``     → retryable
    * ``InternalServerError``    → retryable
    * ``AuthenticationError``    → fatal
    * ``BadRequestError``        → fatal
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError("pip install memcord[openai] to use OpenAI backend")
        super().__init__(
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.model = model
        self.client = AsyncOpenAI()

    async def _ask_impl(self, prompt: str, system: str | None = None) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        content = resp.choices[0].message.content
        return content or ""

    # ── classify subclass-specific errors ──────────────────────────────

    def _is_retryable(self, error: Exception) -> bool:
        if isinstance(error, RetryableError):
            return True

        # OpenAI SDK errors (accessible even without the import guard
        # because they only appear at runtime when the SDK is installed).
        if _OPENAI_AVAILABLE:
            # Rate limits and transient server errors → retry
            if isinstance(error, openai.RateLimitError):
                return True
            if isinstance(error, openai.APITimeoutError):
                return True
            if isinstance(error, openai.APIConnectionError):
                return True
            if isinstance(error, openai.InternalServerError):
                return True
            # Auth / bad requests → do NOT retry
            if isinstance(error, openai.AuthenticationError):
                return False
            if isinstance(error, openai.BadRequestError):
                return False

        return super()._is_retryable(error)
