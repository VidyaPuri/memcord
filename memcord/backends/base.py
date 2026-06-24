"""Base backend with shared retry, timeout, and circuit-breaker logic."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from abc import abstractmethod

from memcord.backends import LLMBackend

logger = logging.getLogger("memcord.backends")


# ── custom error taxonomy ──────────────────────────────────────────────


class RetryableError(Exception):
    """An error that should trigger a retry (transient)."""


class FatalError(Exception):
    """An error that should NOT be retried (permanent)."""


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open — fail-fast."""


# ── shared base class ─────────────────────────────────────────────────


class BaseBackend(LLMBackend):
    """Every concrete backend inherits from this.

    Provides:
    * configurable retry with exponential backoff
    * configurable per-request timeout
    * error classification (retryable vs fatal)
    * circuit-breaker that opens after N consecutive failures
    * structured logging
    """

    def __init__(
        self,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
        circuit_break_seconds: float = 30.0,
        circuit_break_threshold: int = 5,
    ) -> None:
        # Env-var overrides with sensible defaults
        self.timeout = timeout if timeout is not None else int(os.getenv("MEMCORD_TIMEOUT", "60"))
        self.max_retries = (
            max_retries if max_retries is not None else int(os.getenv("MEMCORD_RETRY_MAX", "3"))
        )
        self.retry_delay = (
            retry_delay
            if retry_delay is not None
            else float(os.getenv("MEMCORD_RETRY_DELAY", "1.0"))
        )
        self.circuit_break_seconds = circuit_break_seconds
        self.circuit_break_threshold = circuit_break_threshold

        # Internal state
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    # ── subclass hook ──────────────────────────────────────────────

    @abstractmethod
    async def _ask_impl(self, prompt: str, system: str | None = None) -> str:
        """Backend-specific implementation.

        Subclasses must override this.  The public ``ask()`` method wraps it
        with retries, timeouts, and circuit-breaker logic.
        """
        ...

    # ── retry classifier ───────────────────────────────────────────

    def _is_retryable(self, error: Exception) -> bool:
        """Return True if *error* is safe to retry.

        Subclasses should call ``super()._is_retryable(error)`` and then
        add any backend-specific checks.  By default only ``RetryableError``
        instances are considered retryable.
        """
        return isinstance(error, RetryableError)

    # ── circuit breaker ────────────────────────────────────────────

    def _reset_circuit(self) -> None:
        """Call on a successful request to reset the failure counter."""
        if self._consecutive_failures > 0:
            logger.info(
                "%s: circuit reset after %d failures",
                type(self).__name__,
                self._consecutive_failures,
            )
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0

    def _record_failure(self) -> None:
        """Record a failure and open the circuit if the threshold is hit."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_break_threshold:
            self._circuit_open_until = time.monotonic() + self.circuit_break_seconds
            logger.error(
                "%s: circuit breaker OPEN for %.1fs (%d consecutive failures)",
                type(self).__name__,
                self.circuit_break_seconds,
                self._consecutive_failures,
            )

    # ── public API ─────────────────────────────────────────────────

    async def ask(self, prompt: str, system: str | None = None) -> str:
        """Send a prompt with retries, timeouts, and circuit-breaker protection."""

        # Circuit breaker check (fail-fast)
        remaining = self._circuit_open_until - time.monotonic()
        if remaining > 0:
            raise CircuitBreakerOpenError(f"Circuit breaker open — {remaining:.1f}s remaining")

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                result = await asyncio.wait_for(
                    self._ask_impl(prompt, system),
                    timeout=self.timeout,
                )
                self._reset_circuit()
                return result

            except asyncio.TimeoutError:
                last_error = RetryableError(f"Request timed out after {self.timeout}s")
                logger.warning(
                    "%s: timeout on attempt %d/%d (limit=%ds)",
                    type(self).__name__,
                    attempt + 1,
                    self.max_retries,
                    self.timeout,
                )

            except Exception as exc:
                last_error = exc
                if self._is_retryable(exc):
                    logger.warning(
                        "%s: retryable error on attempt %d/%d: %s",
                        type(self).__name__,
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                else:
                    # Fatal — don't retry, but still count it against the
                    # circuit breaker so a genuinely broken backend opens fast.
                    self._record_failure()
                    logger.error(
                        "%s: fatal error (no retry): %s",
                        type(self).__name__,
                        exc,
                    )
                    raise

            # Exponential backoff before the next attempt
            if attempt < self.max_retries - 1:
                delay = self.retry_delay * (2**attempt)
                logger.debug(
                    "%s: retrying in %.1fs ...",
                    type(self).__name__,
                    delay,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        self._record_failure()
        raise FatalError(
            f"{type(self).__name__}: all {self.max_retries} attempts failed. "
            f"Last error: {last_error}"
        ) from last_error
