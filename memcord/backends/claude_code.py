"""Claude Code CLI backend — calls ``claude -p`` via subprocess."""

from __future__ import annotations

import asyncio
import json
import logging
import os

from memcord.backends.base import BaseBackend, RetryableError

logger = logging.getLogger("memcord.backends.claude_code")


class ClaudeCodeBackend(BaseBackend):
    """Backend that shells out to the ``claude`` CLI (print mode)."""

    def __init__(
        self,
        model: str = "sonnet",
        max_turns: int = 3,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        super().__init__(
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        self.model = model
        self.max_turns = max_turns

    async def _ask_impl(self, prompt: str, system: str | None = None) -> str:
        """Run the Claude CLI and return the result text."""
        if system:
            prompt = f"{system}\n\n---\n\n{prompt}"

        cmd = [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "json",
            "--max-turns",
            str(self.max_turns),
            "--model",
            self.model,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                **os.environ,
                "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
            },
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RetryableError(f"Claude CLI exited {proc.returncode}: {stderr.decode()}")

        try:
            data = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RetryableError(f"Claude CLI returned unparseable JSON: {exc}") from exc

        if data.get("subtype") != "success":
            raise RetryableError(f"Claude CLI error subtype={data.get('subtype', 'unknown')}")

        return data["result"]

    # ── classify subclass-specific errors ──────────────────────────────

    def _is_retryable(self, error: Exception) -> bool:
        # Subprocess failures, JSON parse errors, and CLI errors are all
        # transient enough to retry.
        if isinstance(error, RetryableError):
            return True
        # Inherit base clasification as well.
        return super()._is_retryable(error)
