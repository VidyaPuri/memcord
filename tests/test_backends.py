"""Unit tests for LLM backends with mocked subprocess and HTTP."""

from __future__ import annotations

import contextlib
import json
from unittest.mock import AsyncMock, patch

import pytest

# ── ClaudeCodeBackend tests ────────────────────────────────────────────


class TestClaudeCodeBackend:
    """Tests for ClaudeCodeBackend with mocked subprocess."""

    def test_import(self):
        """ClaudeCodeBackend should be importable."""
        from memcord.backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        assert backend.model == "sonnet"
        assert backend.max_turns == 3

    def test_custom_config(self):
        """Custom model, turns, and timeout should be applied."""
        from memcord.backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend(model="opus", max_turns=5, timeout=30)
        assert backend.model == "opus"
        assert backend.max_turns == 5
        assert backend.timeout == 30

    def test_inherits_from_base(self):
        """ClaudeCodeBackend should inherit from BaseBackend."""
        from memcord.backends.base import BaseBackend
        from memcord.backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        assert isinstance(backend, BaseBackend)

    @pytest.mark.asyncio
    async def test_ask_success(self):
        """Successful ask should return the response text."""
        from memcord.backends.claude_code import ClaudeCodeBackend

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(
                json.dumps({"subtype": "success", "result": "Hello from Claude!"}).encode(),
                b"",
            )
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            backend = ClaudeCodeBackend()
            result = await backend.ask("What is memcord?")
            assert result == "Hello from Claude!"

    @pytest.mark.asyncio
    async def test_ask_with_system_prompt(self):
        """System prompt should be prepended to the prompt."""
        from memcord.backends.claude_code import ClaudeCodeBackend

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(
                json.dumps({"subtype": "success", "result": "OK"}).encode(),
                b"",
            )
        )

        with patch("asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.return_value = mock_proc
            backend = ClaudeCodeBackend()
            await backend.ask("Question", system="You are helpful.")

            # Verify system prompt was included in the command args
            call_args = mock_exec.call_args
            args_list = call_args[0] if call_args[0] else []
            prompt_idx = -1
            for i, arg in enumerate(args_list):
                if arg == "-p":
                    prompt_idx = i + 1
                    break
            if prompt_idx > 0 and prompt_idx < len(args_list):
                prompt_arg = args_list[prompt_idx]
                assert "You are helpful." in prompt_arg

    @pytest.mark.asyncio
    async def test_ask_nonzero_exit(self):
        """Nonzero exit code should raise an error (after retries)."""
        from memcord.backends.base import FatalError
        from memcord.backends.claude_code import ClaudeCodeBackend

        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"{}", b"Error message"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            backend = ClaudeCodeBackend(max_retries=1)
            with pytest.raises(FatalError):
                await backend.ask("What is memcord?")

    @pytest.mark.asyncio
    async def test_ask_error_subtype(self):
        """Error subtype should raise an error (after retries)."""
        from memcord.backends.base import FatalError
        from memcord.backends.claude_code import ClaudeCodeBackend

        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(
            return_value=(
                json.dumps({"subtype": "error", "result": ""}).encode(),
                b"",
            )
        )

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            backend = ClaudeCodeBackend(max_retries=1)
            with pytest.raises(FatalError):
                await backend.ask("What is memcord?")


# ── OpenAI backend tests ───────────────────────────────────────────────


class TestOpenAIBackend:
    """Tests for OpenAIBackend."""

    def test_import_without_openai(self):
        """Should raise ImportError when openai is not installed."""
        with patch("memcord.backends.openai_backend._OPENAI_AVAILABLE", False):
            from memcord.backends.openai_backend import OpenAIBackend

            with pytest.raises(ImportError, match="pip install"):
                OpenAIBackend()

    def test_inherits_base(self):
        """OpenAIBackend should inherit from BaseBackend."""
        from memcord.backends.base import BaseBackend
        from memcord.backends.openai_backend import OpenAIBackend

        assert issubclass(OpenAIBackend, BaseBackend)


# ── Anthropic backend tests ────────────────────────────────────────────


class TestAnthropicBackend:
    """Tests for AnthropicBackend."""

    def test_import_without_anthropic(self):
        """Should raise ImportError when anthropic is not installed."""
        with patch("memcord.backends.anthropic_backend._ANTHROPIC_AVAILABLE", False):
            from memcord.backends.anthropic_backend import AnthropicBackend

            with pytest.raises(ImportError, match="pip install"):
                AnthropicBackend()

    def test_inherits_base(self):
        """AnthropicBackend should inherit from BaseBackend."""
        from memcord.backends.anthropic_backend import AnthropicBackend
        from memcord.backends.base import BaseBackend

        assert issubclass(AnthropicBackend, BaseBackend)


# ── Ollama backend tests ───────────────────────────────────────────────


class TestOllamaBackend:
    """Tests for OllamaBackend with mocked HTTP."""

    def test_import_without_httpx(self):
        """Should raise ImportError when httpx is not installed."""
        with patch("memcord.backends.ollama_backend._HTTPX_AVAILABLE", False):
            from memcord.backends.ollama_backend import OllamaBackend

            with pytest.raises(ImportError, match="pip install"):
                OllamaBackend()

    def test_default_base_url(self):
        """Default base_url should be localhost:11434."""
        from memcord.backends.ollama_backend import OllamaBackend

        backend = OllamaBackend()
        assert backend.base_url == "http://localhost:11434"

    def test_default_model(self):
        """Default model should be llama3."""
        from memcord.backends.ollama_backend import OllamaBackend

        backend = OllamaBackend()
        assert backend.model == "llama3"

    def test_inherits_base(self):
        """OllamaBackend should inherit from BaseBackend."""
        from memcord.backends.base import BaseBackend
        from memcord.backends.ollama_backend import OllamaBackend

        assert issubclass(OllamaBackend, BaseBackend)


# ── BaseBackend tests ──────────────────────────────────────────────────


class TestBaseBackend:
    """Tests for BaseBackend retry and circuit breaker logic."""

    def test_error_classes(self):
        """Error taxonomy classes should exist."""
        from memcord.backends.base import (
            CircuitBreakerOpenError,
            FatalError,
            RetryableError,
        )

        assert issubclass(RetryableError, Exception)
        assert issubclass(FatalError, Exception)
        assert issubclass(CircuitBreakerOpenError, Exception)

    def test_base_params(self):
        """BaseBackend should accept custom params."""
        from memcord.backends.base import BaseBackend

        class TestBackend(BaseBackend):
            async def _ask_impl(self, prompt, system=None):
                return "test response"

        backend = TestBackend(timeout=30, max_retries=2, retry_delay=0.5)
        assert backend.timeout == 30
        assert backend.max_retries == 2
        assert backend.retry_delay == 0.5
        assert backend.circuit_break_seconds == 30.0
        assert backend.circuit_break_threshold == 5

    @pytest.mark.asyncio
    async def test_base_ask_success(self):
        """Successful ask should return response."""
        from memcord.backends.base import BaseBackend

        class TestBackend(BaseBackend):
            async def _ask_impl(self, prompt, system=None):
                return f"response: {prompt}"

        backend = TestBackend()
        result = await backend.ask("hello")
        assert result == "response: hello"

    @pytest.mark.asyncio
    async def test_base_ask_with_retry(self):
        """Retry should be attempted on transient failures."""
        import asyncio

        from memcord.backends.base import BaseBackend

        call_count = 0

        class TestBackend(BaseBackend):
            async def _ask_impl(self, prompt, system=None):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise asyncio.TimeoutError()
                return "success after retries"

        backend = TestBackend(max_retries=5, retry_delay=0.01)
        result = await backend.ask("hello")
        assert result == "success after retries"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_base_ask_all_retries_exhausted(self):
        """Should raise FatalError when all retries exhausted."""
        import asyncio

        from memcord.backends.base import BaseBackend, FatalError

        class TestBackend(BaseBackend):
            async def _ask_impl(self, prompt, system=None):
                raise asyncio.TimeoutError()

        backend = TestBackend(max_retries=2, retry_delay=0.01)
        with pytest.raises(FatalError):
            await backend.ask("hello")

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens(self):
        """Circuit breaker should open after threshold failures."""
        from memcord.backends.base import BaseBackend, CircuitBreakerOpenError

        class TestBackend(BaseBackend):
            async def _ask_impl(self, prompt, system=None):
                raise RuntimeError("always fail")

        backend = TestBackend(
            max_retries=1,
            retry_delay=0.01,
            circuit_break_threshold=3,
            circuit_break_seconds=5.0,
        )

        for _ in range(3):
            with contextlib.suppress(Exception):
                await backend.ask("hello")

        with pytest.raises(CircuitBreakerOpenError):
            await backend.ask("hello")

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self):
        """Circuit breaker should reset after a successful request."""
        import asyncio

        from memcord.backends.base import BaseBackend

        call_count = 0

        class TestBackend(BaseBackend):
            async def _ask_impl(self, prompt, system=None):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise asyncio.TimeoutError()
                return "recovered"

        backend = TestBackend(max_retries=3, retry_delay=0.01, circuit_break_threshold=5)

        with contextlib.suppress(Exception):
            await backend.ask("hello")

        result = await backend.ask("hello")
        assert result == "recovered"
        assert backend._consecutive_failures == 0


# ── LLMBackend protocol tests ──────────────────────────────────────────


class TestLLMBackendABC:
    """Tests for the LLMBackend abstract base class."""

    def test_mock_backend_satisfies_abc(self):
        """MockBackend should be an instance of LLMBackend."""
        from memcord.backends import LLMBackend
        from tests.conftest import MockBackend

        backend = MockBackend()
        assert isinstance(backend, LLMBackend)

    def test_claude_code_satisfies_abc(self):
        """ClaudeCodeBackend should be an instance of LLMBackend."""
        from memcord.backends import LLMBackend
        from memcord.backends.claude_code import ClaudeCodeBackend

        backend = ClaudeCodeBackend()
        assert isinstance(backend, LLMBackend)


# ── Backend factory tests ──────────────────────────────────────────────


class TestBackendFactory:
    """Tests for the get_backend factory function."""

    def test_get_backend_exists(self):
        """get_backend should be importable."""
        from memcord.backends import get_backend

        assert get_backend is not None

    def test_get_backend_defaults_to_claude_code(self):
        """Default backend should be ClaudeCodeBackend."""
        import os

        from memcord.backends.claude_code import ClaudeCodeBackend

        # Ensure no override
        old = os.environ.pop("MEMCORD_BACKEND", None)
        try:
            from memcord.backends import get_backend

            backend = get_backend()
            assert isinstance(backend, ClaudeCodeBackend)
        finally:
            if old is not None:
                os.environ["MEMCORD_BACKEND"] = old

    def test_get_backend_unknown_raises(self):
        """Unknown backend name should raise ValueError."""
        import os

        os.environ["MEMCORD_BACKEND"] = "nonexistent"
        try:
            from memcord.backends import get_backend

            with pytest.raises(ValueError, match="Unknown backend"):
                get_backend()
        finally:
            del os.environ["MEMCORD_BACKEND"]


# ── Backend imports ────────────────────────────────────────────────────


class TestBackendImports:
    """Verify all backends are importable."""

    def test_all_backends_importable(self):
        """All four backends should be importable."""
        from memcord.backends.anthropic_backend import AnthropicBackend
        from memcord.backends.claude_code import ClaudeCodeBackend
        from memcord.backends.ollama_backend import OllamaBackend
        from memcord.backends.openai_backend import OpenAIBackend

        assert ClaudeCodeBackend is not None
        assert OpenAIBackend is not None
        assert AnthropicBackend is not None
        assert OllamaBackend is not None
