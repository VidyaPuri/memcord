"""Health check HTTP endpoint — runs on a separate port alongside the Discord bot."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memcord.discord_.bot import MemcordBot

log = logging.getLogger("memcord.health")

_HEALTH_PORT: int | None = None
_HEALTH_SERVER: asyncio.AbstractServer | None = None


def _make_health_handler(bot: MemcordBot):
    """Return an aiohttp-style request handler that serves health status."""
    from aiohttp import web

    async def handler(request: web.Request) -> web.Response:
        stats = bot.cache.stats
        body = {
            "status": "ok",
            "cache_hits": stats["cache_hits"],
            "total_queries": stats["total_queries"],
            "hit_rate": stats["hit_rate"],
            "cached_faqs": stats["cached_faqs"],
            "threshold": stats["threshold"],
            "adaptive": stats["adaptive"],
        }
        return web.json_response(body)

    return handler


async def start_health_server(bot: MemcordBot, port: int) -> None:
    """Start an aiohttp health check server on *port*.

    The server shuts down gracefully when the event loop is stopped.
    """
    global _HEALTH_SERVER, _HEALTH_PORT
    if _HEALTH_SERVER is not None:
        log.warning("Health server already running — ignoring duplicate start")
        return

    from aiohttp import web

    app = web.Application()
    app.router.add_get("/health", _make_health_handler(bot))
    app.router.add_get("/", _make_health_handler(bot))  # also serve at root

    runner = web.AppRunner(app)
    await runner.setup()

    try:
        server = web.TCPSite(runner, "0.0.0.0", port)
        await server.start()
    except OSError:
        await runner.cleanup()
        raise

    _HEALTH_SERVER = server
    _HEALTH_PORT = port
    log.info(f"Health check listening on http://0.0.0.0:{port}/health")


async def stop_health_server() -> None:
    """Shut down the health server if running."""
    global _HEALTH_SERVER, _HEALTH_PORT
    if _HEALTH_SERVER is not None:
        await _HEALTH_SERVER.stop()
        _HEALTH_SERVER = None
        _HEALTH_PORT = None
        log.info("Health server stopped")


def get_health_port() -> int | None:
    """Return the port the health server is listening on, or None."""
    return _HEALTH_PORT
