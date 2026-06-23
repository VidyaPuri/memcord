"""CLI entry point — `memcord run`."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("memcord")


def _get_backend():
    """Instantiate backend based on env vars."""
    backend_name = os.getenv("MEMCORD_BACKEND", "claude_code")
    model = os.getenv("MEMCORD_MODEL", "")

    if backend_name == "claude_code":
        from memcord.backends.claude_code import ClaudeCodeBackend
        return ClaudeCodeBackend(model=model or "sonnet")

    if backend_name == "openai":
        from memcord.backends.openai_backend import OpenAIBackend
        return OpenAIBackend(model=model or "gpt-4o")

    if backend_name == "anthropic":
        from memcord.backends.anthropic_backend import AnthropicBackend
        return AnthropicBackend(model=model or "claude-sonnet-4-20250514")

    if backend_name == "ollama":
        from memcord.backends.ollama_backend import OllamaBackend
        return OllamaBackend(model=model or "llama3")

    if backend_name == "custom":
        import importlib
        path = os.getenv("MEMCORD_CUSTOM_BACKEND", "")
        if not path:
            raise ValueError("MEMCORD_CUSTOM_BACKEND must be set (e.g. 'mypackage.MyBackend')")
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)()

    raise ValueError(f"Unknown backend: {backend_name}")


def run():
    """Start the Memcord bot."""
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        log.error("DISCORD_TOKEN not set in .env")
        sys.exit(1)

    from memcord.cache import FAQCache
    from memcord.discord_ import MemcordBot

    cache = FAQCache(
        data_dir=os.getenv("MEMCORD_DATA_DIR", "./memcord_data"),
        similarity_threshold=float(os.getenv("MEMCORD_SIMILARITY_THRESHOLD", "0.80")),
        adaptive_threshold=os.getenv("MEMCORD_ADAPTIVE_THRESHOLD", "true").lower() != "false",
    )

    backend = _get_backend()
    log.info(f"Backend: {type(backend).__name__}")
    log.info(f"Cache: threshold={cache.threshold}, adaptive={cache.adaptive}")

    bot = MemcordBot(cache=cache, backend=backend)
    bot.run(token)


def main():
    parser = argparse.ArgumentParser(description="Memcord — self-learning Discord FAQ bot")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("run", help="Start the Discord bot")
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "version":
        from memcord import __version__
        print(f"memcord v{__version__}")
    elif args.command == "run":
        run()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
