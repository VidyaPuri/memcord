"""Ollama backend — local LLM via Ollama REST API."""

from __future__ import annotations

import os

from memcord.backends import LLMBackend

try:
    import httpx
except ImportError:
    httpx = None


class OllamaBackend:
    """Backend for Ollama's local REST API."""

    def __init__(self, model: str = "llama3", base_url: str | None = None):
        if httpx is None:
            raise ImportError("pip install memcord[ollama] to use Ollama backend")
        self.model = model
        self.base_url = base_url or os.getenv("MEMCORD_OLLAMA_URL", "http://localhost:11434")

    async def ask(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]
