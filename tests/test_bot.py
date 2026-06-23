"""Unit tests for MemcordBot — rate limiting, response chunking, sanitization."""

from __future__ import annotations

import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ── Helper functions (standalone testable logic) ───────────────────────


def sanitize_mention(content: str, user_id: int) -> str:
    """Remove bot mention from message content (extracted from on_message)."""
    return content.replace(f"<@{user_id}>", "").strip()


def chunk_response(text: str, max_length: int = 2000) -> list[str]:
    """Split a long response into Discord-message-sized chunks.

    Tries to split on paragraph or sentence boundaries.
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    current = ""

    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_length:
            if current:
                current += "\n\n" + para
            else:
                current = para
        else:
            if current:
                chunks.append(current)
            # If a single paragraph is too long, split by sentences
            if len(para) > max_length:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current = ""
                for sent in sentences:
                    if len(current) + len(sent) + 1 <= max_length:
                        if current:
                            current += " " + sent
                        else:
                            current = sent
                    else:
                        if current:
                            chunks.append(current)
                        current = sent
            else:
                current = para

    if current:
        chunks.append(current)

    return chunks or [text]


class SimpleRateLimiter:
    """Simple per-user rate limiter for testing."""

    def __init__(self, max_requests: int = 5, window_seconds: float = 10.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._users: dict[int, list[float]] = {}
        self._cooldowns: dict[int, float] = {}

    def check(self, user_id: int, now: float | None = None) -> bool:
        """Check if user is allowed to make a request. Returns True if allowed."""
        import time as _time

        now = now or _time.monotonic()

        # Check cooldown
        if user_id in self._cooldowns:
            if now < self._cooldowns[user_id]:
                return False
            del self._cooldowns[user_id]

        # Clean old entries
        if user_id not in self._users:
            self._users[user_id] = []

        cutoff = now - self.window_seconds
        self._users[user_id] = [t for t in self._users[user_id] if t > cutoff]

        if len(self._users[user_id]) >= self.max_requests:
            self._cooldowns[user_id] = now + self.window_seconds
            return False

        self._users[user_id].append(now)
        return True

    def reset_user(self, user_id: int) -> None:
        """Reset rate limit state for a user."""
        self._users.pop(user_id, None)
        self._cooldowns.pop(user_id, None)


# ── Sanitization tests ─────────────────────────────────────────────────


class TestSanitization:
    """Tests for message content sanitization."""

    def test_remove_mention(self):
        """Bot mention should be stripped from content."""
        content = "<@123456789> How do I install memcord?"
        result = sanitize_mention(content, 123456789)
        assert result == "How do I install memcord?"

    def test_remove_mention_only(self):
        """Message that is just a mention should become empty."""
        content = "<@123456789>"
        result = sanitize_mention(content, 123456789)
        assert result == ""

    def test_remove_mention_with_extra_whitespace(self):
        """Leading/trailing whitespace should be stripped."""
        content = "   <@123456789>   help me   "
        result = sanitize_mention(content, 123456789)
        assert result == "help me"

    def test_no_mention_present(self):
        """Content without mention should be unchanged."""
        content = "How do I install memcord?"
        result = sanitize_mention(content, 123456789)
        assert result == "How do I install memcord?"

    def test_different_user_id(self):
        """Mention with different user ID should not be removed."""
        content = "<@999999999> How do I install?"
        result = sanitize_mention(content, 123456789)
        assert "<@999999999>" in result

    def test_multiple_mentions(self):
        """All instances of the bot mention should be removed."""
        content = "<@123> hello <@123> world"
        result = sanitize_mention(content, 123)
        assert result == "hello world"


# ── Response chunking tests ────────────────────────────────────────────


class TestResponseChunking:
    """Tests for response chunking (Discord 2000 char limit)."""

    def test_short_response_no_chunking(self):
        """Short response should return single chunk."""
        text = "This is a short response."
        chunks = chunk_response(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_exactly_at_limit(self):
        """Response exactly at 2000 chars should be one chunk."""
        text = "x" * 2000
        chunks = chunk_response(text)
        assert len(chunks) == 1
        assert len(chunks[0]) == 2000

    def test_barely_over_limit(self):
        """Response just over 2000 should be split."""
        text = "x" * 2001
        chunks = chunk_response(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert len(c) <= 2000

    def test_long_response_multi_chunk(self):
        """Response of ~6000 chars should produce multiple chunks."""
        text = "Hello world. " * 500
        chunks = chunk_response(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 2000

    def test_paragraph_boundary_split(self):
        """Prefer splitting on paragraph boundaries."""
        para1 = "First paragraph. " * 100
        para2 = "Second paragraph. " * 100
        text = para1 + "\n\n" + para2
        chunks = chunk_response(text)
        assert len(chunks) >= 2

    def test_empty_response(self):
        """Empty string should return list with empty string."""
        chunks = chunk_response("")
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_all_chunks_non_empty(self):
        """No chunk should be empty after splitting."""
        text = "A. " * 1500
        chunks = chunk_response(text)
        for c in chunks:
            assert len(c) > 0


# ── Rate limiting tests ────────────────────────────────────────────────


class TestRateLimiter:
    """Tests for SimpleRateLimiter."""

    def test_first_request_allowed(self):
        """First request should always be allowed."""
        limiter = SimpleRateLimiter(max_requests=5)
        assert limiter.check(1, now=0.0) is True

    def test_within_limit(self):
        """Requests within the limit should be allowed."""
        limiter = SimpleRateLimiter(max_requests=5)
        for i in range(5):
            assert limiter.check(1, now=float(i)) is True

    def test_over_limit_blocked(self):
        """Requests over the limit should be blocked."""
        limiter = SimpleRateLimiter(max_requests=3)
        for i in range(3):
            assert limiter.check(1, now=float(i)) is True
        assert limiter.check(1, now=3.0) is False

    def test_window_expires(self):
        """After window expires, user should be allowed again."""
        limiter = SimpleRateLimiter(max_requests=3, window_seconds=10.0)
        for i in range(3):
            assert limiter.check(1, now=float(i)) is True
        assert limiter.check(1, now=3.0) is False
        assert limiter.check(1, now=20.0) is True

    def test_different_users_independent(self):
        """Rate limits should be per-user."""
        limiter = SimpleRateLimiter(max_requests=2)
        assert limiter.check(1, now=0.0) is True
        assert limiter.check(1, now=1.0) is True
        assert limiter.check(1, now=2.0) is False

        assert limiter.check(2, now=2.0) is True
        assert limiter.check(2, now=3.0) is True

    def test_reset_user(self):
        """Resetting a user should clear their limits."""
        limiter = SimpleRateLimiter(max_requests=2)
        limiter.check(1, now=0.0)
        limiter.check(1, now=1.0)
        assert limiter.check(1, now=2.0) is False

        limiter.reset_user(1)
        assert limiter.check(1, now=3.0) is True

    def test_cooldown_period(self):
        """After exceeding limit, user enters cooldown."""
        limiter = SimpleRateLimiter(max_requests=2, window_seconds=5.0)
        limiter.check(1, now=0.0)
        limiter.check(1, now=1.0)
        assert limiter.check(1, now=2.0) is False
        assert limiter.check(1, now=4.0) is False
        assert limiter.check(1, now=8.0) is True


# ── MockBackend tests (via fixtures) ───────────────────────────────────


class TestMockBackendIntegration:
    """Tests for MockBackend used as a bot LLM backend."""

    @pytest.mark.asyncio
    async def test_mock_ask_returns_response(self, mock_backend):
        """MockBackend.ask() should return a predictable response."""
        mock_backend.set_responses("Test response")
        result = await mock_backend.ask("What is memcord?")
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_mock_tracks_calls(self, mock_backend):
        """MockBackend should track all ask() calls."""
        await mock_backend.ask("Question 1")
        await mock_backend.ask("Question 2", system="Be helpful")
        await mock_backend.ask("Question 3")

        assert mock_backend.call_count == 3
        assert mock_backend.calls[0]["prompt"] == "Question 1"
        assert mock_backend.calls[1]["system"] == "Be helpful"
        assert mock_backend.calls[2]["prompt"] == "Question 3"

    @pytest.mark.asyncio
    async def test_mock_simulates_failure(self, mock_backend):
        """MockBackend should simulate failures for error path testing."""
        mock_backend.set_responses("first ok")
        mock_backend.set_failure(RuntimeError("API down"), after_calls=1)

        result = await mock_backend.ask("hello")
        assert result == "first ok"

        with pytest.raises(RuntimeError, match="API down"):
            await mock_backend.ask("hello again")

    @pytest.mark.asyncio
    async def test_mock_reset(self, mock_backend):
        """MockBackend.reset() should clear all state."""
        await mock_backend.ask("Q1")
        assert mock_backend.call_count == 1

        mock_backend.reset()
        assert mock_backend.call_count == 0
        assert mock_backend.calls == []

    def test_mock_backend_is_llm_backend(self, mock_backend):
        """MockBackend should satisfy the LLMBackend ABC."""
        from memcord.backends import LLMBackend
        assert isinstance(mock_backend, LLMBackend)


# ── MemcordBot import tests ────────────────────────────────────────────


class TestMemcordBotStructure:
    """Basic structure tests for the bot module."""

    def test_bot_importable(self):
        """MemcordBot should be importable."""
        from memcord.discord_.bot import MemcordBot

        assert MemcordBot is not None

    def test_system_prompt_exists(self):
        """System prompt constant should exist."""
        from memcord.discord_.bot import SYSTEM_PROMPT

        assert isinstance(SYSTEM_PROMPT, str)
        assert len(SYSTEM_PROMPT) > 0

    def test_commands_defined(self):
        """All slash commands should be defined."""
        from memcord.discord_.bot import (
            faq_add,
            faq_adaptive,
            faq_remove,
            faq_stats,
            faq_threshold,
            listen,
            stop,
        )

        assert listen is not None
        assert stop is not None
        assert faq_stats is not None
        assert faq_add is not None
        assert faq_remove is not None
        assert faq_threshold is not None
        assert faq_adaptive is not None
