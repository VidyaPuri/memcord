"""Minimal smoke test for the backend refactoring — bypasses ChromaDB."""
import sys, os

# Test 1: Imports
from memcord.backends import LLMBackend, get_backend
from memcord.backends.base import (
    BaseBackend, RetryableError, FatalError, CircuitBreakerOpenError
)
print("Test 1 OK: all base imports successful")

# Test 2: LLMBackend is an ABC
from abc import ABC
assert issubclass(LLMBackend, ABC), "LLMBackend must be an ABC"
print("Test 2 OK: LLMBackend is ABC")

# Test 3: BaseBackend inherits from LLMBackend
assert issubclass(BaseBackend, LLMBackend), "BaseBackend must inherit from LLMBackend"
print("Test 3 OK: BaseBackend inherits from LLMBackend")

# Test 4: All 4 concrete backends inherit from BaseBackend
from memcord.backends.claude_code import ClaudeCodeBackend
from memcord.backends.openai_backend import OpenAIBackend
from memcord.backends.anthropic_backend import AnthropicBackend
from memcord.backends.ollama_backend import OllamaBackend

for cls in [ClaudeCodeBackend, OpenAIBackend, AnthropicBackend, OllamaBackend]:
    assert issubclass(cls, BaseBackend), f"{cls.__name__} must inherit from BaseBackend"
print("Test 4 OK: all 4 backends inherit from BaseBackend")

# Test 5: Each backend implements _ask_impl (abstract method)
import inspect
for cls in [ClaudeCodeBackend, OpenAIBackend, AnthropicBackend, OllamaBackend]:
    assert hasattr(cls, '_ask_impl'), f"{cls.__name__} missing _ask_impl"
    assert callable(getattr(cls, '_ask_impl')), f"{cls.__name__}._ask_impl not callable"
print("Test 5 OK: all backends have _ask_impl")

# Test 6: get_backend factory function
assert callable(get_backend), "get_backend must be callable"
print("Test 6 OK: get_backend is callable")

# Test 7: Instantiate backends (no real API calls)
os.environ["MEMCORD_TIMEOUT"] = "30"
os.environ["MEMCORD_RETRY_MAX"] = "2"
os.environ["MEMCORD_RETRY_DELAY"] = "0.5"

# ClaudeCodeBackend doesn't need API key just to construct
cc = ClaudeCodeBackend(timeout=45)
assert cc.timeout == 45, f"timeout should be 45, got {cc.timeout}"
assert cc.max_retries == 2, f"max_retries should be 2 (env), got {cc.max_retries}"
assert cc.retry_delay == 0.5, f"retry_delay should be 0.5 (env), got {cc.retry_delay}"
print("Test 7 OK: ClaudeCodeBackend construction + env var overrides")

# Test 8: OllamaBackend (no httpx needed for construction)
try:
    from memcord.backends.ollama_backend import OllamaBackend
    ollama = OllamaBackend(timeout=30, max_retries=1)
    assert ollama.model == "llama3"
    print("Test 8 OK: OllamaBackend construction")
except ImportError:
    print("Test 8 SKIP: httpx not installed (expected in CI)")

# Test 9: _ask_impl is abstract on BaseBackend
try:
    base = BaseBackend()
    assert False, "Should not be able to instantiate BaseBackend directly"
except TypeError as e:
    assert "abstract" in str(e).lower()
    print("Test 9 OK: BaseBackend cannot be instantiated (abstract)")

# Test 10: Circuit breaker state
cc2 = ClaudeCodeBackend()
assert cc2._consecutive_failures == 0
assert cc2._circuit_open_until == 0.0
cc2._record_failure()
assert cc2._consecutive_failures == 1
cc2._reset_circuit()
assert cc2._consecutive_failures == 0
print("Test 10 OK: circuit breaker state machine")

# Test 11: Error classification (RetryableError)
assert BaseBackend._is_retryable(cc2, RetryableError("test"))
assert not BaseBackend._is_retryable(cc2, ValueError("test"))
print("Test 11 OK: error classification")

# Test 12: CLI uses factory
from memcord.cli import _get_backend
os.environ["MEMCORD_BACKEND"] = "claude_code"
os.environ["MEMCORD_MODEL"] = "sonnet"
backend_from_cli = _get_backend()
assert isinstance(backend_from_cli, ClaudeCodeBackend)
print("Test 12 OK: CLI _get_backend delegates to factory")

print("\n=== ALL BACKEND REFACTORING TESTS PASSED ===")
