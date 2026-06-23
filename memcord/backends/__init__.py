"""LLM backends — pluggable, protocol-driven."""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    """Every backend must implement this."""

    async def ask(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt, return the response text."""
        ...
