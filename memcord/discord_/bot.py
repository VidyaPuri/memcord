"""Discord bot — wires everything together."""

from __future__ import annotations

import logging

import discord
from discord.ext import commands

from memcord.cache import FAQCache
from memcord.backends import LLMBackend

log = logging.getLogger("memcord.bot")

# ── system prompt ──────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant in a Discord server. Keep answers concise.
If you don't know something, say so honestly. Don't make things up."""


# ── bot ────────────────────────────────────────────────────

class MemcordBot(commands.Bot):
    """The Memcord Discord bot."""

    def __init__(
        self,
        cache: FAQCache,
        backend: LLMBackend,
        system_prompt: str = SYSTEM_PROMPT,
        *args,
        **kwargs,
    ):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        kwargs.setdefault("command_prefix", "/")
        kwargs.setdefault("intents", intents)
        super().__init__(*args, **kwargs)

        self.cache = cache
        self.backend = backend
        self.system_prompt = system_prompt

    async def setup_hook(self) -> None:
        self.add_command(listen)
        self.add_command(stop)
        self.add_command(faq_stats)
        self.add_command(faq_add)
        self.add_command(faq_remove)
        self.add_command(faq_threshold)
        log.info("MemcordBot commands registered")

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

        # Clean the message
        content = message.content.replace(f"<@{self.user.id}>", "").strip()
        if not content:
            await message.reply("Yes?")
            return

        # 1. Check FAQ cache
        cached, similarity = self.cache.check(content)
        if cached:
            log.info(f"FAQ HIT similarity={similarity:.3f} — \"{content[:60]}...\"")
            reply = await message.reply(f"{cached}\n\n-# 💾 FAQ cache • similarity {similarity:.0%}")
            await reply.add_reaction("👍")
            await reply.add_reaction("👎")
            return

        # 2. Cache miss — call LLM
        log.info(f"FAQ MISS — \"{content[:60]}...\"")
        async with message.channel.typing():
            try:
                response = await self.backend.ask(content, system=self.system_prompt)
            except Exception as e:
                log.error(f"LLM error: {e}")
                await message.reply(f"Sorry, the LLM backend returned an error: {e}")
                return

        # 3. Store in cache
        self.cache.store(content, response)

        # 4. Reply with feedback buttons
        reply = await message.reply(response)
        await reply.add_reaction("👍")
        await reply.add_reaction("👎")

        await self.process_commands(message)

    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User) -> None:
        if user.bot:
            return
        if reaction.emoji not in ("👍", "👎"):
            return

        msg = reaction.message
        # Only track reactions on the bot's own messages
        if msg.author != self.user:
            return

        # Only track the first reaction per user
        for r in msg.reactions:
            if r.emoji in ("👍", "👎") and r.emoji != reaction.emoji:
                async for u in r.users():
                    if u == user:
                        return

        # Find which question this was answering
        if not msg.reference or not msg.reference.resolved:
            return
        question = msg.reference.resolved.content
        if not question:
            return

        self.cache.feedback(question, positive=(reaction.emoji == "👍"))
        log.info(f"Feedback {'+' if reaction.emoji == '👍' else '-'} on FAQ: \"{question[:50]}...\"")


# ── slash commands ─────────────────────────────────────────

@commands.command(name="listen", aliases=["li"])
async def listen(ctx: commands.Context) -> None:
    """Start listening to @mentions in this server."""
    # Memcord always listens to @mentions — this is a no-op for UX
    await ctx.send("I'm always listening for @mentions! Just tag me or reply to my messages.")


@commands.command(name="stop", aliases=["s"])
async def stop(ctx: commands.Context) -> None:
    """Stop the bot."""
    await ctx.send("Use Ctrl+C in the terminal or `/shutdown` to stop the bot.")


@commands.command(name="faq-stats", aliases=["faq"])
async def faq_stats(ctx: commands.Context) -> None:
    """Show FAQ cache statistics."""
    bot = ctx.bot
    if not isinstance(bot, MemcordBot):
        return
    s = bot.cache.stats
    await ctx.send(
        f"**FAQ Cache Stats**\n"
        f"• Cached FAQs: {s['cached_faqs']}\n"
        f"• Total queries: {s['total_queries']}\n"
        f"• Cache hits: {s['cache_hits']}\n"
        f"• Hit rate: {s['hit_rate']:.1%}\n"
        f"• Threshold: {s['threshold']}\n"
        f"• Adaptive: {'on' if s['adaptive'] else 'off'}"
    )


@commands.command(name="faq-add")
async def faq_add(ctx: commands.Context, *, text: str = "") -> None:
    """Manually add a FAQ. Format: /faq-add question | answer"""
    if "|" not in text:
        await ctx.send("Usage: `/faq-add question | answer`")
        return
    question, answer = text.split("|", 1)
    bot = ctx.bot
    if isinstance(bot, MemcordBot):
        bot.cache.store(question.strip(), answer.strip())
        await ctx.send(f"FAQ added: \"{question.strip()[:80]}...\"")


@commands.command(name="faq-remove")
async def faq_remove(ctx: commands.Context) -> None:
    """Remove a FAQ by reacting 👎 3 times on its answer."""
    await ctx.send("To remove a FAQ, react with 👎 on its answer message 3 times. It'll be auto-removed.")


@commands.command(name="faq-threshold")
async def faq_threshold(ctx: commands.Context, value: float = 0) -> None:
    """Set the FAQ cache similarity threshold (0.0 - 1.0)."""
    if not 0.0 < value <= 1.0:
        await ctx.send("Threshold must be between 0.0 and 1.0. Example: `/faq-threshold 0.85`")
        return
    bot = ctx.bot
    if isinstance(bot, MemcordBot):
        bot.cache.adaptive = False
        bot.cache.threshold = value
        await ctx.send(f"FAQ threshold set to {value:.2f} (adaptive mode disabled)")


@commands.command(name="faq-adaptive")
async def faq_adaptive(ctx: commands.Context) -> None:
    """Re-enable adaptive threshold mode."""
    bot = ctx.bot
    if isinstance(bot, MemcordBot):
        bot.cache.adaptive = True
        await ctx.send(f"Adaptive threshold enabled. Current threshold: {bot.cache.threshold:.3f}")
