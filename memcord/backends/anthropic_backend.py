"""Anthropic API backend (direct SDK, no CLI needed)."""

from __future__ import annotations

import logging

from memcord.backends.base import BaseBackend, RetryableError

logger = logging.getLogger("memcord.backends.anthropic")

try:
    import anthropic
    from anthropic import AsyncAnthropic

    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
    AsyncAnthropic = None  # type: ignore[assignment,misc]


class AnthropicBackend(BaseBackend):
    """Backend using the Anthropic Python SDK.

    Handles:
    * ``RateLimitError``      → retryable
    * ``OverloadedError``     → retryable
    * ``APITimeoutError``     → retryable
    * ``APIConnectionError``  → retryable
    * ``InternalServerError`` → retryable
    * ``AuthenticationError`` → fatal
    * ``BadRequestError``     → fatal
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("pip install memcord[anthropic] to use Anthropic backend")
        super().__init__(
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.model = model
        self.client = AsyncAnthropic()

    async def _ask_impl(self, prompt: str, system: str | None = None) -> str:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": 2048,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        resp = await self.client.messages.create(**kwargs)
        return resp.content[0].text

    # ── classify subclass-specific errors ──────────────────────────────

    def _is_retryable(self, error: Exception) -> bool:
        if isinstance(error, RetryableError):
            return True

        if _ANTHROPIC_AVAILABLE:
            # Rate limits and server-side overloads → retry
            if isinstance(error, anthropic.RateLimitError):
                return True
            if isinstance(error, anthropic.OverloadedError):
                return True
            if isinstance(error, anthropic.APITimeoutError):
                return True
            if isinstance(error, anthropic.APIConnectionError):
                return True
            if isinstance(error, anthropic.InternalServerError):
                return True
            # Auth / bad requests → do NOT retry
            if isinstance(error, anthropic.AuthenticationError):
                return False
            if isinstance(error, anthropic.BadRequestError):
                return False

        return super()._is_retryable(error)
