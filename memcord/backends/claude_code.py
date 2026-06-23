"""Claude Code CLI backend — calls `claude -p` via subprocess."""

from __future__ import annotations

import asyncio
import json
import os

from memcord.backends import LLMBackend


class ClaudeCodeBackend:
    """Backend that shells out to `claude` CLI in print mode."""

    def __init__(self, model: str = "sonnet", max_turns: int = 3, timeout: int = 60):
        self.model = model
        self.max_turns = max_turns
        self.timeout = timeout

    async def ask(self, prompt: str, system: str | None = None) -> str:
        if system:
            prompt = f"{system}\n\n---\n\n{prompt}"

        cmd = [
            "claude", "-p", prompt,
            "--output-format", "json",
            "--max-turns", str(self.max_turns),
            "--model", self.model,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "")},
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            raise RuntimeError(f"Claude Code timed out after {self.timeout}s")

        if proc.returncode != 0:
            raise RuntimeError(f"Claude Code failed (exit {proc.returncode}): {stderr.decode()}")

        data = json.loads(stdout)
        if data.get("subtype") != "success":
            raise RuntimeError(f"Claude Code error: {data.get('subtype')}")

        return data["result"]
