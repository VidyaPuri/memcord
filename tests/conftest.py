"""Shared test fixtures and MockBackend for memcord tests."""

from __future__ import annotations

import shutil
from typing import Any

import pytest

from memcord.backends import LLMBackend
from memcord.cache import FAQCache

# ── MockBackend ────────────────────────────────────────────────────────


class MockBackend(LLMBackend):
    """Mock LLM backend for testing.

    Returns predictable responses, tracks calls for assertions,
    and can simulate failures for error path testing.
    """

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize with optional preset responses.

        Args:
            responses: List of responses to return on successive calls.
                       If None, returns a default response.
        """
        self._responses = responses or []
        self._idx = 0
        self.calls: list[dict[str, Any]] = []
        self._simulate_failure: Exception | None = None
        self._failure_after: int = -1  # fail after N successful calls

    async def ask(self, prompt: str, system: str | None = None) -> str:
        """Mock LLMBackend.ask() implementation."""
        call = {"prompt": prompt, "system": system}
        self.calls.append(call)

        # Simulate failure if configured
        if self._failure_after >= 0 and len(self.calls) > self._failure_after:
            if self._simulate_failure:
                raise self._simulate_failure
            raise RuntimeError("MockBackend: simulated failure")

        if self._responses and self._idx < len(self._responses):
            response = self._responses[self._idx]
            self._idx += 1
            return response

        return f"Mock response to: {prompt[:50]}"

    def set_responses(self, *responses: str) -> None:
        """Set a sequence of responses to return."""
        self._responses = list(responses)
        self._idx = 0

    def set_failure(self, exception: Exception, after_calls: int = 0) -> None:
        """Configure the mock to raise an exception.

        Args:
            exception: The exception to raise.
            after_calls: Number of successful calls before failure.
        """
        self._simulate_failure = exception
        self._failure_after = after_calls

    def reset(self) -> None:
        """Reset all internal state."""
        self._responses = []
        self._idx = 0
        self.calls = []
        self._simulate_failure = None
        self._failure_after = -1

    @property
    def call_count(self) -> int:
        """Number of times ask() was called."""
        return len(self.calls)


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Provide a temporary directory for FAQCache data."""
    d = tmp_path / "memcord_test_data"
    d.mkdir()
    yield str(d)
    shutil.rmtree(str(d), ignore_errors=True)


@pytest.fixture
def cache(tmp_cache_dir):
    """Create a fresh FAQCache instance for testing."""
    return FAQCache(data_dir=tmp_cache_dir)


@pytest.fixture
def mock_backend():
    """Create a MockBackend instance."""
    return MockBackend()


@pytest.fixture
def backend_with_responses():
    """Create a MockBackend with preset responses."""

    def _make(*responses: str) -> MockBackend:
        return MockBackend(responses=list(responses))

    return _make
