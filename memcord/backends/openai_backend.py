"""OpenAI API backend."""

from __future__ import annotations

from memcord.backends import LLMBackend

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None


class OpenAIBackend:
    """Backend using the OpenAI Python SDK."""

    def __init__(self, model: str = "gpt-4o"):
        if AsyncOpenAI is None:
            raise ImportError("pip install memcord[openai] to use OpenAI backend")
        self.model = model
        self.client = AsyncOpenAI()

    async def ask(self, prompt: str, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages
        )
        return resp.choices[0].message.content
