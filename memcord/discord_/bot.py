"""Discord bot — wires everything together with rate limiting, chunking, sanitization, and conversation support."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import signal
import time
from collections import defaultdict, deque

import discord
from discord.ext import commands

from memcord.backends import LLMBackend
from memcord.cache import FAQCache
from memcord.metrics import Metrics
from memcord.security import check_prompt_injection

log = logging.getLogger("memcord.bot")

# ── config from env ────────────────────────────────────────

RATE_LIMIT = int(os.getenv("MEMCORD_RATE_LIMIT", "5"))
RATE_WINDOW = int(os.getenv("MEMCORD_RATE_WINDOW", "60"))
CONTEXT_MESSAGES = int(os.getenv("MEMCORD_CONTEXT_MESSAGES", "5"))
CHUNK_SIZE = 1900  # safe below Discord's 2000-char limit

# ── system prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant in a Discord server. Keep answers concise.
If you don't know something, say so honestly. Don't make things up."""

# ── mention regex ──────────────────────────────────────────

_MENTION_RE = re.compile(r"<@!?\d+>")


# ── helpers ────────────────────────────────────────────────


def _sanitize_input(content: str) -> str:
    """Strip all @mentions, limit length, handle empty/whitespace-only."""
    # Strip all @mentions
    content = _MENTION_RE.sub("", content)
    # Limit to MEMCORD_MAX_INPUT_LENGTH (default 2000)
    max_len = int(os.getenv("MEMCORD_MAX_INPUT_LENGTH", "2000"))
    content = content[:max_len]
    # Normalise whitespace
    content = content.strip()
    return content


def _chunk_text(text: str, size: int = CHUNK_SIZE) -> list[str]:
    """Split text into chunks at sentence/paragraph boundaries, each ≤ size chars."""
    if len(text) <= size:
        return [text]

    chunks = []
    while len(text) > size:
        # Try to split at the last sentence end within the chunk
        split_at = size
        for sep in ("\n\n", "\n", ". ", "! ", "? ", " "):
            pos = text.rfind(sep, 0, size)
            if pos > size // 2:  # only use if it's a reasonable split
                split_at = pos + len(sep)
                break
        else:
            # Fallback: hard split at size
            split_at = size

        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    if text:
        chunks.append(text)

    return chunks


# ── bot ────────────────────────────────────────────────────


class MemcordBot(commands.Bot):
    """The Memcord Discord bot."""

    # Class attribute so commands can check without isinstance
    is_memcord: bool = True

    def __init__(
        self,
        cache: FAQCache,
        backend: LLMBackend,
        metrics: Metrics | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        *args,
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        kwargs.setdefault("command_prefix", "!")
        kwargs.setdefault("intents", intents)
        super().__init__(*args, **kwargs)

        self.cache = cache
        self.backend = backend
        self.metrics = metrics or Metrics()
        self.system_prompt = system_prompt

        # Wire metrics into the cache for deletion tracking
        if hasattr(self.cache, "set_metrics"):
            self.cache.set_metrics(self.metrics)

        # ── cache serialization ─────────────────────────────
        # FAQCache operations (ChromaDB + SentenceBERT) are synchronous
        # and block the asyncio event loop. Offload them to a thread pool
        # and serialize with a lock to prevent ChromaDB race conditions.
        self._cache_lock = asyncio.Lock()

        # ── rate limiting ──────────────────────────────────
        # per-user: deque of timestamps
        self._rate_limits: dict[int, deque[float]] = defaultdict(
            lambda: deque(maxlen=RATE_LIMIT * 2)
        )

        # ── conversation context ───────────────────────────
        # per-channel: deque of (author_name, content) tuples
        self._context: dict[int, deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=CONTEXT_MESSAGES)
        )

        # ── shutdown flag ──────────────────────────────────
        self._shutting_down = False

        # ── graceful shutdown ──────────────────────────────
        self._setup_signal_handlers()

    # ── signal handling ────────────────────────────────────

    def _setup_signal_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers for graceful shutdown."""
        with contextlib.suppress(RuntimeError):
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                with contextlib.suppress(NotImplementedError, RuntimeError):
                    loop.add_signal_handler(sig, self._handle_shutdown_signal)

    def _handle_shutdown_signal(self) -> None:
        """Schedule graceful shutdown."""
        if self._shutting_down:
            return
        self._shutting_down = True
        log.info("Shutdown signal received — closing gracefully…")
        asyncio.create_task(self._graceful_shutdown())

    async def _graceful_shutdown(self) -> None:
        """Flush stats, close ChromaDB, then close the bot."""
        try:
            self.cache.flush_stats()
            log.info("Stats flushed to disk")
        except Exception as e:
            log.error(f"Error flushing stats: {e}")
        await self.close()

    async def close(self) -> None:
        """Override close to ensure stats and metrics are saved before exit."""
        try:
            self.cache.flush_stats()
        except Exception as e:
            log.error(f"Error flushing stats on close: {e}")
        try:
            self.metrics.stop_periodic_logging()
            self.metrics.export_json()
            log.info("Final metrics exported on shutdown")
        except Exception as e:
            log.error(f"Error exporting metrics on close: {e}")
        await super().close()

    # ── Discord event hooks ────────────────────────────────

    async def setup_hook(self) -> None:
        self.add_command(listen)
        self.add_command(stop)
        self.add_command(faq_stats)
        self.add_command(faq_metrics)
        self.add_command(faq_list)
        self.add_command(faq_add)
        self.add_command(faq_remove)
        self.add_command(faq_threshold)
        self.add_command(faq_adaptive)
        log.info("MemcordBot commands registered")

        # Start periodic metrics logging (every 5 min)
        self._metrics_task = await self.metrics.start_periodic_logging(interval=300)

    async def on_ready(self) -> None:
        log.info(f"Logged in as {self.user} (ID: {self.user.id})")

    async def on_message(self, message: discord.Message) -> None:
        if message.author.bot:
            return

        # Only respond to @mentions or replies to the bot
        mentioned = self.user in message.mentions
        is_reply_to_bot = (
            message.reference
            and message.reference.resolved
            and message.reference.resolved.author == self.user
        )
        if not (mentioned or is_reply_to_bot):
            await self.process_commands(message)
            return

        # ── rate limiting ──────────────────────────────────
        if not self._check_rate_limit(message.author.id):
            log.warning(f"Rate limit hit for user {message.author} ({message.author.id})")
            await message.reply(
                f"⏳ You're asking too fast — please wait {RATE_WINDOW}s between batches of {RATE_LIMIT} questions."
            )
            return

        # ── input sanitization ─────────────────────────────
        content = _sanitize_input(message.content)
        # Remove the bot's @mention (already stripped by sanitizer, but handle edge cases)
        bot_mention = f"<@{self.user.id}>"
        content = content.replace(bot_mention, "").strip()
        bot_mention_alt = f"<@!{self.user.id}>"
        content = content.replace(bot_mention_alt, "").strip()

        if not content:
            await message.reply("Yes?")
            return

        # ── security: prompt injection detection ────────────
        if check_prompt_injection(content):
            log.warning(f"Prompt injection detected — blocked message from {message.author}")
            await message.reply("I can't process that request. Please ask a different question.")
            return

        # ── conversation context ───────────────────────────
        # Track this message
        ctx_deque = self._context[message.channel.id]
        ctx_deque.append((str(message.author), content))

        # Build context string if user is replying to the bot
        context_str = ""
        if is_reply_to_bot and len(ctx_deque) > 1:
            # The last entry is the current message; everything before is context
            recent = list(ctx_deque)[:-1]
            if recent:
                context_str = "Recent conversation:\n" + "\n".join(
                    f"{who}: {txt[:300]}" for who, txt in recent[-CONTEXT_MESSAGES:]
                )
                context_str += "\n\n"

        # 1. Check FAQ cache
        async with self._cache_lock:
            cached, similarity = await asyncio.to_thread(self.cache.check, content)
        if cached:
            self.metrics.cache_hit()
            log.info(f'FAQ HIT similarity={similarity:.3f} — "{content[:60]}..."')
            reply = await self._send_chunked(
                message,
                f"{cached}\n\n-# 💾 FAQ cache • similarity {similarity:.0%}",
            )
            if reply:
                await reply.add_reaction("👍")
                await reply.add_reaction("👎")
            return

        # 2. Cache miss — call LLM
        self.metrics.cache_miss()
        log.info(f'FAQ MISS — "{content[:60]}..."')
        async with message.channel.typing():
            try:
                full_prompt = context_str + content if context_str else content
                t0 = time.time()
                response = await self.backend.ask(full_prompt, system=self.system_prompt)
                latency_ms = (time.time() - t0) * 1000
                self.metrics.llm_call(latency_ms)
            except Exception as e:
                self.metrics.llm_error()
                log.error(f"LLM error: {e}")
                await message.reply(f"Sorry, the LLM backend returned an error: {e}")
                return

        # 3. Store in cache
        async with self._cache_lock:
            await asyncio.to_thread(self.cache.store, content, response)

        # 4. Reply with feedback buttons (chunked)
        reply = await self._send_chunked(message, response)
        if reply:
            await reply.add_reaction("👍")
            await reply.add_reaction("👎")

        await self.process_commands(message)

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent) -> None:
        # Ignore self-reactions (works in DMs where payload.member is None)
        if payload.user_id == self.user.id:
            return
        # Ignore bot reactions
        if payload.member and payload.member.bot:
            return
        if str(payload.emoji) not in ("👍", "👎"):
            return

        # Fetch the message (works across restarts)
        channel = self.get_channel(payload.channel_id)
        if channel is None:
            return
        try:
            msg = await channel.fetch_message(payload.message_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return

        # Only track reactions on the bot's own messages
        if msg.author != self.user:
            return

        # Find which question this was answering
        if not msg.reference or not msg.reference.resolved:
            return
        question = msg.reference.resolved.content
        if not question:
            return

        # One vote per user — check if they already reacted with the other emoji
        for r in msg.reactions:
            if str(r.emoji) not in ("👍", "👎"):
                continue
            if str(r.emoji) == str(payload.emoji):
                continue
            async for u in r.users():
                if u.id == payload.user_id:
                    return  # already voted

        positive = str(payload.emoji) == "👍"
        async with self._cache_lock:
            await asyncio.to_thread(self.cache.feedback, question, positive=positive)
        if positive:
            self.metrics.feedback_up()
        else:
            self.metrics.feedback_down()
        log.info(f'Feedback {"+" if positive else "-"} on FAQ: "{question[:50]}..."')

    # ── rate limiting ──────────────────────────────────────

    def _check_rate_limit(self, user_id: int) -> bool:
        """Check if user has exceeded rate limit. Returns True if allowed."""
        now = time.time()
        timestamps = self._rate_limits[user_id]

        # Expire old entries
        cutoff = now - RATE_WINDOW
        while timestamps and timestamps[0] < cutoff:
            timestamps.popleft()

        if len(timestamps) >= RATE_LIMIT:
            return False

        timestamps.append(now)
        return True

    # ── response chunking ──────────────────────────────────

    async def _send_chunked(self, message: discord.Message, text: str) -> discord.Message | None:
        """Send a potentially long response as multiple messages. Returns the first message sent."""
        chunks = _chunk_text(text)
        if not chunks:
            return None

        first = None
        for i, chunk in enumerate(chunks):
            try:
                if i == 0:
                    # First chunk: reply to the original message
                    msg = await message.reply(chunk)
                    first = msg
                else:
                    # Subsequent chunks: send as follow-ups in the channel
                    msg = await message.channel.send(chunk)
                    if first is None:
                        first = msg
            except discord.HTTPException as e:
                log.error(f"Failed to send chunk {i}: {e}")
                break
            # Small delay to avoid rate limiting on chunked sends
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)

        return first


# ── helper to access the bot ───────────────────────────────


def _get_bot(ctx: commands.Context) -> MemcordBot | None:
    """Safely get the MemcordBot instance without isinstance."""
    bot = ctx.bot
    if getattr(bot, "is_memcord", False):
        return bot  # type: ignore[return-value]
    return None


# ── prefix commands (!) ──────────────────────────────────────


@commands.command(name="listen", aliases=["li"])
async def listen(ctx: commands.Context) -> None:
    """Start listening to @mentions in this server."""
    await ctx.send("I'm always listening for @mentions! Just tag me or reply to my messages.")


@commands.command(name="stop", aliases=["s"])
async def stop(ctx: commands.Context) -> None:
    """Stop the bot."""
    await ctx.send("Use Ctrl+C in the terminal to stop the bot.")


@commands.command(name="faq-stats", aliases=["faq"])
async def faq_stats(ctx: commands.Context) -> None:
    """Show FAQ cache statistics."""
    bot = _get_bot(ctx)
    if not bot:
        return
    s = bot.cache.stats
    await ctx.send(
        f"**FAQ Cache Stats**\n"
        f"• Cached FAQs: {s['cached_faqs']}\n"
        f"• Total queries: {s['total_queries']}\n"
        f"• Cache hits: {s['cache_hits']}\n"
        f"• Hit rate: {s['hit_rate']:.1%}\n"
        f"• Threshold: {s['threshold']}\n"
        f"• Adaptive: {'on' if s['adaptive'] else 'off'}\n"
        f"• 👍 {s['upvotes']}  👎 {s['downvotes']}  🗑 {s['deletions']}  ➕ {s['stores']}"
    )


@commands.command(name="faq-list")
async def faq_list(ctx: commands.Context, query: str = "") -> None:
    """Show recent FAQs or search for specific ones. Usage: !faq-list [search term]"""
    bot = _get_bot(ctx)
    if not bot:
        return

    try:
        faqs = bot.cache.list_faqs(query=query if query else None, limit=10)
    except Exception as e:
        await ctx.send(f"Error listing FAQs: {e}")
        return

    if not faqs:
        await ctx.send("No FAQs found.")
        return

    lines = [f"**{'Search results' if query else 'Recent FAQs'}** ({len(faqs)} found)\n"]
    for i, faq in enumerate(faqs, 1):
        q = faq["question"][:80]
        a = faq["answer"][:120]
        rating = faq.get("rating", 0)
        sim = faq.get("similarity")
        created = faq.get("created_at", 0)
        rating_str = f"{rating:+d}" if rating else "0"

        line = f"**{i}.** Q: {q}"
        if sim is not None:
            line += f" `{sim:.0%}`"
        line += f" [{rating_str}]"
        line += f"\n　 A: {a}"
        if created:
            when = time.strftime("%Y-%m-%d", time.localtime(created))
            line += f" _({when})_"
        lines.append(line)

    # Send possibly chunked
    text = "\n".join(lines)
    if len(text) > CHUNK_SIZE:
        for chunk in _chunk_text(text):
            await ctx.send(chunk)
    else:
        await ctx.send(text)


@commands.command(name="faq-add")
async def faq_add(ctx: commands.Context, *, text: str = "") -> None:
    """Manually add a FAQ. Format: !faq-add question | answer"""
    if "|" not in text:
        await ctx.send("Usage: `!faq-add question | answer`")
        return
    question, answer = text.split("|", 1)
    bot = _get_bot(ctx)
    if bot:
        bot.cache.store(question.strip(), answer.strip())
        await ctx.send(f'FAQ added: "{question.strip()[:80]}..."')


@commands.command(name="faq-remove")
async def faq_remove(ctx: commands.Context) -> None:
    """Remove a FAQ by reacting 👎 3 times on its answer."""
    await ctx.send(
        "To remove a FAQ, react with 👎 on its answer message 3 times. It'll be auto-removed."
    )


@commands.command(name="faq-threshold")
async def faq_threshold(ctx: commands.Context, value: float = 0) -> None:
    """Set the FAQ cache similarity threshold (0.0 - 1.0)."""
    if not 0.0 < value <= 1.0:
        await ctx.send("Threshold must be between 0.0 and 1.0. Example: `!faq-threshold 0.85`")
        return
    bot = _get_bot(ctx)
    if bot:
        bot.cache.adaptive = False
        bot.cache.threshold = value
        await ctx.send(f"FAQ threshold set to {value:.2f} (adaptive mode disabled)")


@commands.command(name="faq-adaptive")
async def faq_adaptive(ctx: commands.Context) -> None:
    """Re-enable adaptive threshold mode."""
    bot = _get_bot(ctx)
    if bot:
        bot.cache.adaptive = True
        await ctx.send(f"Adaptive threshold enabled. Current threshold: {bot.cache.threshold:.3f}")


@commands.command(name="faq-metrics", aliases=["metrics"])
async def faq_metrics(ctx: commands.Context) -> None:
    """Show structured bot metrics: LLM calls, latency, cache stats, feedback."""
    bot = _get_bot(ctx)
    if not bot:
        return
    m = bot.metrics.snapshot()
    await ctx.send(
        f"**📊 Bot Metrics**\n"
        f"• Uptime: {m['uptime_seconds']:.0f}s\n"
        f"• LLM calls: {m['llm_calls']}  •  errors: {m['llm_errors']}\n"
        f"• LLM latency: avg {m['llm_latency_avg_ms']}ms, "
        f"recent {m['llm_latency_recent_avg_ms']}ms, "
        f"min {m['llm_latency_min_ms']}ms, "
        f"max {m['llm_latency_max_ms']}ms\n"
        f"• Cache hits: {m['cache_hits']}  •  misses: {m['cache_misses']}  "
        f" •  hit rate: {m['cache_hit_rate']:.1%}\n"
        f"• Feedback: 👍{m['feedback_up']}  👎{m['feedback_down']}\n"
        f"• FAQ deletions: {m['faq_deletions']}"
    )
