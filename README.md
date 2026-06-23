# Memcord

Self-learning Discord FAQ bot with semantic caching. Listens to questions, learns which ones repeat, and serves cached answers without calling the LLM — regardless of which LLM backend you use.

## The Problem

Every AI Discord bot calls the LLM API for every single message. When 50 people ask "how do I install this?", you pay for 50 API calls and users wait 2-10 seconds each time.

## The Solution

Memcord intercepts questions, checks if a semantically similar question has been answered before, and returns the cached answer instantly. Only genuinely new questions hit the LLM.

## Architecture

```
Discord message → FAQ Cache (ChromaDB) → [HIT: return cached] [MISS: LLM → cache → return]
                                                         ↑
                                              Feedback loop (👍👎)
```

## Features

- **Semantic FAQ cache**: ChromaDB + SentenceBERT embeddings
- **Adaptive threshold**: auto-tunes similarity cutoff based on hit rate
- **LLM-agnostic**: Claude Code, OpenAI, Anthropic, Ollama, or custom
- **Feedback loop**: 👍👎 reactions improve the cache
- **Admin commands**: /faq-stats, /faq-add, /faq-remove, /faq-threshold
- **Zero config startup**: `pip install -e . && memcord run`

## Quick Start

```bash
git clone https://github.com/yourusername/memcord
cd memcord
pip install -e .
cp .env.example .env  # add DISCORD_TOKEN + LLM keys
memcord run
```

## License

MIT
