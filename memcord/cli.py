"""CLI entry point — `memcord run`."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
import sys

from dotenv import load_dotenv

load_dotenv()

# ── logging ─────────────────────────────────────────────────
# Respect MEMCORD_LOG_LEVEL env var (default: INFO)
_log_level_name = os.getenv("MEMCORD_LOG_LEVEL", "INFO").upper()
_log_level = getattr(logging, _log_level_name, logging.INFO)
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("memcord")


# ── token redaction ─────────────────────────────────────────

def _redact_token(token: str) -> str:
    """Redact a Discord token for safe logging: show only first 6 and last 4 chars."""
    if not token:
        return "<not set>"
    if len(token) <= 10:
        return token[:3] + "***"
    return token[:6] + "..." + token[-4:]


_TOKEN = os.getenv("DISCORD_TOKEN", "")


def _get_backend():
    """Instantiate backend based on env vars — delegates to the factory."""
    from memcord.backends import get_backend
    return get_backend()


def _get_health_port() -> int:
    """Resolve health check port from env or default."""
    return int(os.getenv("MEMCORD_HEALTH_PORT", "8080"))


async def _async_run(health_port: int, token: str, metrics_enabled: bool = True) -> None:
    """Async entry point — start health server, then the Discord bot."""
    from memcord.cache import FAQCache
    from memcord.discord_ import MemcordBot
    from memcord.health import start_health_server, stop_health_server
    from memcord.metrics import Metrics
    from memcord.model_presets import resolve_model

    cache = FAQCache(
        data_dir=os.getenv("MEMCORD_DATA_DIR", "./memcord_data"),
        model_name=resolve_model(os.getenv("MEMCORD_EMBEDDING_MODEL", "fast")),
        similarity_threshold=float(os.getenv("MEMCORD_SIMILARITY_THRESHOLD", "0.80")),
        adaptive_threshold=os.getenv("MEMCORD_ADAPTIVE_THRESHOLD", "true").lower() != "false",
        feedback_threshold=float(os.getenv("MEMCORD_FEEDBACK_THRESHOLD", "0.80")),
    )
    log.info(f"Embedding model: {cache._model_name}")
    backend = _get_backend()
    log.info(f"Backend: {type(backend).__name__}")
    log.info(f"Cache: threshold={cache.threshold}, adaptive={cache.adaptive}")

    metrics = Metrics() if metrics_enabled else None
    if metrics:
        log.info(f"Metrics: enabled, export={metrics._export_path}")
    else:
        log.info("Metrics: disabled")

    bot = MemcordBot(cache=cache, backend=backend, metrics=metrics)

    # Start health check server if port > 0
    if health_port > 0:
        try:
            await start_health_server(bot, health_port)
        except OSError as e:
            log.error(f"Failed to start health server on port {health_port}: {e}")
            sys.exit(1)

    # Graceful shutdown handler
    async def shutdown() -> None:
        log.info("Shutting down...")
        if health_port > 0:
            await stop_health_server()
        await bot.close()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: asyncio.ensure_future(shutdown()))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await bot.start(token)
    except Exception:
        log.error(
            "Failed to start bot (token: %s). Check that your Discord token is valid.",
            _redact_token(token),
        )
        await shutdown()
        sys.exit(1)


def run(health_port: int | None = None, metrics: bool = True) -> None:
    """Start the Memcord bot."""
    from memcord.security import validate_env

    # Validate environment before anything else
    log.info("Validating environment variables...")
    validate_env()
    log.info("Environment validation passed ✓")

    if not _TOKEN:
        log.error("DISCORD_TOKEN not set in .env")
        sys.exit(1)

    if health_port is None:
        health_port = _get_health_port()

    log.info(f"Health check port: {health_port if health_port > 0 else 'disabled'}")
    log.info(f"Discord token: {_redact_token(_TOKEN)}")
    log.info(f"Metrics: {'enabled' if metrics else 'disabled'}")
    asyncio.run(_async_run(health_port, _TOKEN, metrics_enabled=metrics))


def main():
    parser = argparse.ArgumentParser(description="Memcord — self-learning Discord FAQ bot")
    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Start the Discord bot")
    run_parser.add_argument(
        "--health-port",
        type=int,
        default=None,
        help="Port for health check HTTP endpoint (env: MEMCORD_HEALTH_PORT, default: 8080, 0=disable)",
    )
    run_parser.add_argument(
        "--metrics",
        action="store_true",
        default=True,
        help="Enable structured metrics collection and periodic logging (default: enabled)",
    )
    run_parser.add_argument(
        "--no-metrics",
        action="store_false",
        dest="metrics",
        help="Disable structured metrics collection",
    )
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        from memcord import __version__
        print(f"memcord v{__version__}")
    elif args.command == "run":
        run(health_port=args.health_port, metrics=args.metrics)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
