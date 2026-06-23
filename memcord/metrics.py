"""Structured observability: counters, latencies, periodic logging, JSON export."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

log = logging.getLogger("memcord.metrics")


class Metrics:
    """Thread-safe structured metrics tracker.

    Tracks:
    - LLM: calls, errors, latency (avg, min, max, recent samples)
    - Cache: hits, misses, hit rate
    - Feedback: upvotes, downvotes
    - FAQ deletions
    - Periodic logging to structured log line every N seconds
    - JSON export to a configurable file path

    Usage::

        m = Metrics()
        m.cache_hit()
        m.llm_call(150.3)
        print(m.snapshot())
    """

    def __init__(self, export_path: str | None = None):
        self._lock = threading.Lock()

        # ── LLM counters ──────────────────────────────────────
        self._llm_calls: int = 0
        self._llm_errors: int = 0
        self._llm_latency_sum_ms: float = 0.0
        self._llm_latency_min_ms: float = float("inf")
        self._llm_latency_max_ms: float = 0.0
        self._llm_latency_samples: list[float] = []  # rolling window (last 100)

        # ── Cache counters ────────────────────────────────────
        self._cache_hits: int = 0
        self._cache_misses: int = 0

        # ── Feedback counters ─────────────────────────────────
        self._feedback_up: int = 0
        self._feedback_down: int = 0

        # ── Deletion counter ──────────────────────────────────
        self._faq_deletions: int = 0

        # ── Export ────────────────────────────────────────────
        self._export_path = export_path or os.getenv(
            "MEMCORD_METRICS_FILE", "./memcord_data/metrics.json"
        )

        # ── Periodic logging ──────────────────────────────────
        self._periodic_task: asyncio.Task | None = None
        self._periodic_interval: int = 300  # 5 minutes

        # ── Uptime tracking ───────────────────────────────────
        self._start_time: float = time.time()

    # ── public emitters ───────────────────────────────────────

    def llm_call(self, latency_ms: float) -> None:
        """Record a successful LLM call with its latency in milliseconds."""
        with self._lock:
            self._llm_calls += 1
            self._llm_latency_sum_ms += latency_ms
            self._llm_latency_samples.append(latency_ms)
            # Rolling window: keep last 100 samples
            if len(self._llm_latency_samples) > 100:
                self._llm_latency_samples.pop(0)
            if latency_ms < self._llm_latency_min_ms:
                self._llm_latency_min_ms = latency_ms
            if latency_ms > self._llm_latency_max_ms:
                self._llm_latency_max_ms = latency_ms

    def llm_error(self) -> None:
        """Record an LLM call that resulted in an error."""
        with self._lock:
            self._llm_errors += 1

    def cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_hits += 1

    def cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1

    def feedback_up(self) -> None:
        """Record a thumbs-up feedback."""
        with self._lock:
            self._feedback_up += 1

    def feedback_down(self) -> None:
        """Record a thumbs-down feedback."""
        with self._lock:
            self._feedback_down += 1

    def faq_deletion(self) -> None:
        """Record an FAQ deletion."""
        with self._lock:
            self._faq_deletions += 1

    # ── snapshot ──────────────────────────────────────────────

    def snapshot(self) -> dict:
        """Return a dict with all current metric values."""
        with self._lock:
            calls = self._llm_calls
            avg_latency = self._llm_latency_sum_ms / max(calls, 1)
            samples = self._llm_latency_samples
            recent_avg = (
                sum(samples) / max(len(samples), 1) if samples else 0.0
            )
            total_cache = self._cache_hits + self._cache_misses
            return {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "llm_calls": calls,
                "llm_errors": self._llm_errors,
                "llm_latency_avg_ms": round(avg_latency, 1),
                "llm_latency_recent_avg_ms": round(recent_avg, 1),
                "llm_latency_min_ms": (
                    round(self._llm_latency_min_ms, 1) if calls > 0 else 0.0
                ),
                "llm_latency_max_ms": round(self._llm_latency_max_ms, 1),
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "cache_hit_rate": round(
                    self._cache_hits / max(total_cache, 1), 3
                ),
                "feedback_up": self._feedback_up,
                "feedback_down": self._feedback_down,
                "faq_deletions": self._faq_deletions,
            }

    # ── JSON export ───────────────────────────────────────────

    def export_json(self) -> None:
        """Write current metrics snapshot to the JSON export file atomically."""
        try:
            data = self.snapshot()
            data["timestamp"] = time.time()
            path = Path(self._export_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(json.dumps(data, indent=2))
            tmp_path.rename(path)
        except Exception:
            log.exception("Failed to export metrics JSON")

    # ── periodic logging ──────────────────────────────────────

    async def start_periodic_logging(
        self, interval: int = 300
    ) -> asyncio.Task:
        """Start a background task that logs a metrics summary every *interval* seconds.

        The task exports JSON on every tick as well.
        Returns the asyncio Task for lifecycle management.
        """
        self._periodic_interval = interval
        self._periodic_task = asyncio.create_task(self._periodic_loop())
        log.info(
            "Metrics periodic logging started (interval=%ds, export=%s)",
            interval,
            self._export_path,
        )
        return self._periodic_task

    async def _periodic_loop(self) -> None:
        """Run forever: sleep, then log + export."""
        while True:
            try:
                await asyncio.sleep(self._periodic_interval)
            except asyncio.CancelledError:
                log.info("Metrics periodic logging stopped")
                return

            snap = self.snapshot()
            log.info(
                "METRICS | uptime=%ss | LLM: %s calls, %s errs, "
                "avg %sms (min %s max %s) | "
                "Cache: %s hits, %s misses (%.1f%%) | "
                "Feedback: 👍%s 👎%s | Deletions: %s",
                snap["uptime_seconds"],
                snap["llm_calls"],
                snap["llm_errors"],
                snap["llm_latency_avg_ms"],
                snap["llm_latency_min_ms"],
                snap["llm_latency_max_ms"],
                snap["cache_hits"],
                snap["cache_misses"],
                snap["cache_hit_rate"] * 100,
                snap["feedback_up"],
                snap["feedback_down"],
                snap["faq_deletions"],
            )
            self.export_json()

    def stop_periodic_logging(self) -> None:
        """Cancel the periodic logging task if it's running."""
        if self._periodic_task is not None:
            self._periodic_task.cancel()
            self._periodic_task = None
