"""Anthropic API backend (direct SDK, no CLI needed)."""

from __future__ import annotations

from memcord.backends import LLMBackend

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None


class AnthropicBackend:
    """Backend using the Anthropic Python SDK."""

    def __init__(self, model: str = "claude-sonnet-4-20250514"):
        if AsyncAnthropic is None:
            raise ImportError("pip install memcord[anthropic] to use Anthropic backend")
        self.model = model
        self.client = AsyncAnthropic()

    async def ask(self, prompt: str, system: str | None = None) -> str:
        kwargs = {"model": self.model, "max_tokens": 2048, "messages": [{"role": "user", "content": prompt}]}
        if system:
            kwargs["system"] = system

        resp = await self.client.messages.create(**kwargs)
        return resp.content[0].text
