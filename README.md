# Memcord

Self-learning Discord FAQ bot with semantic caching. Listens to questions, learns which ones repeat, and serves cached answers without calling the LLM — regardless of which LLM backend you use.

[![CI](https://github.com/VidyaPuri/memcord/actions/workflows/ci.yml/badge.svg)](https://github.com/VidyaPuri/memcord/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Discord](https://img.shields.io/badge/discord.py-2.3+-5865F2.svg)](https://discordpy.readthedocs.io/)

## The Problem

Every AI Discord bot calls the LLM API for every single message. When 50 people ask "how do I install this?", you pay for 50 API calls and users wait 2-10 seconds each time.

## The Solution

Memcord intercepts questions, checks if a semantically similar question has been answered before, and returns the cached answer instantly. Only genuinely new questions hit the LLM.

```
@memcord how do I install this?
→ FAQ cache hit (similarity 97%) → instant reply, $0 cost

@memcord can you explain quantum computing?
→ FAQ cache miss → Claude Code → answer → cached for next time
```

## Features

- **Semantic FAQ cache**: ChromaDB + SentenceBERT embeddings — matches "how do I install" with "how to install memcord"
- **Adaptive threshold**: auto-tunes similarity cutoff based on real hit rate — no manual tuning
- **LLM-agnostic**: Claude Code, OpenAI, Anthropic, Ollama, or [bring your own](#custom-backends)
- **Feedback loop**: 👍👎 reactions improve the cache; 3 downvotes auto-deletes bad FAQs
- **Pluggable backends**: one Protocol, four implementations, zero lock-in
- **Admin commands**: `!faq-stats`, `!faq-add`, `!faq-list`, `!faq-threshold`, `!faq-adaptive`

## Cache as a Standalone Library

The FAQ cache is now a reusable library — other applications can embed it without pulling in Discord, backends, or the CLI:

```python
from memcord import SemanticAnswerCache, build_cache, CacheHit

# Simple usage
cache = build_cache(data_dir="./my_cache")
cache.observe("how do I install?", "Run: pip install myapp", scope="support")
hit = cache.lookup("how do I install?", scope="support")
if hit:
    print(hit.answer)  # cached!
```

### Features

- **Scoped partitioning**: `scope` is an opaque key — entries under `scope="a"` are never returned for `scope="b"`
- **Caller-driven hooks**:
  - `should_cache(answer) → bool` — gates which answers are stored
  - `validate(CacheHit) → bool` — filters candidates at lookup time
- **Conservative promotion** (`promote_after=N`): entries become retrievable only after being observed N times (default 1 = immediate)
- **Pluggable embedding model** via `build_cache(embed_model=...)`
- **Quality voting**: `cache.vote(hit_id, +1/-1)` with automatic downvote-prune

### Example: Custom Hooks

```python
cache = build_cache(
    data_dir="./cache",
    promote_after=3,                              # only after 3 observations
    should_cache=lambda a: len(a) > 5,            # drop very short answers
    validate=lambda hit: "safe" in hit.answer,     # only serve safe answers
)
```

The bundled Discord bot remains one consumer of this library — it still imports `FAQCache` unchanged.

## Quick Start

```bash
git clone https://github.com/VidyaPuri/memcord
cd memcord
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openai]"  # or [anthropic] or [ollama]

cp .env.example .env
# Edit .env: add DISCORD_TOKEN, OPENAI_API_KEY, set MEMCORD_BACKEND=openai

memcord run
```

### With Claude Code

```bash
npm install -g @anthropic-ai/claude-code
# Set in .env: MEMCORD_BACKEND=claude_code, ANTHROPIC_API_KEY=...
memcord run
```

### With Ollama (fully local)

```bash
ollama pull llama3
# Set in .env: MEMCORD_BACKEND=ollama, MEMCORD_MODEL=llama3
memcord run
```

## Backends

| Backend | Command | Dependencies | API Cost |
|---------|---------|-------------|----------|
| Claude Code | `claude -p` via subprocess | `npm install -g @anthropic-ai/claude-code` | Anthropic |
| OpenAI | Python SDK | `pip install memcord[openai]` | OpenAI |
| Anthropic | Python SDK | `pip install memcord[anthropic]` | Anthropic |
| Ollama | REST API | `pip install memcord[ollama]` | Free (local) |
| Custom | Your own class | — | Your choice |

## Custom Backends

Implement `BaseBackend` to get retry, timeout, and circuit-breaker for free:

```python
# my_backend.py
from memcord.backends.base import BaseBackend

class MyBackend(BaseBackend):
    async def _ask_impl(self, prompt: str, system: str | None = None) -> str:
        return "your response"
```

Then run:
```bash
MEMCORD_BACKEND=custom MEMCORD_CUSTOM_BACKEND=my_backend.MyBackend memcord run
```

## Commands

Memcord uses `!` as its command prefix (to avoid colliding with Discord's native slash commands).

| Command | Description |
|---------|-------------|
| `!faq-stats` | Show cache hit rate, FAQ count, threshold |
| `!faq-metrics` | Show detailed metrics (LLM calls, latency, etc.) |
| `!faq-add Q \| A` | Manually add a FAQ |
| `!faq-list [search]` | List or search FAQs |
| `!faq-threshold 0.85` | Set similarity threshold, disable adaptive |
| `!faq-adaptive` | Re-enable adaptive threshold mode |
| `!faq-remove` | Instructions for removing FAQs via 👎 reactions |

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

```
Discord message → FAQ Cache (ChromaDB) → [HIT: return cached] [MISS: LLM → cache → return]
                                                         ↑
                                              Feedback loop (👍👎)
```

## License

MIT — see [LICENSE](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md)
