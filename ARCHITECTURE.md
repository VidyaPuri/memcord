# Architecture

```
┌──────────────────────────────────────────────────┐
│                   DISCORD SERVER                  │
│   User → @memcord "how do I install?"             │
└────────────────────┬─────────────────────────────┘
                     │ WebSocket (discord.py)
                     ▼
┌──────────────────────────────────────────────────┐
│                 MEMCORD BOT                        │
│  ┌─────────────┐  ┌────────────┐  ┌────────────┐ │
│  │ Rate Limiter│  │ Sanitizer  │  │ Chunker    │ │
│  │ (per-user)  │  │ (mentions) │  │ (>1900→✂️) │ │
│  └──────┬──────┘  └─────┬──────┘  └─────┬──────┘ │
│         └───────────────┼───────────────┘        │
│                         ▼                         │
│              ┌──────────────────┐                 │
│              │   FAQ CACHE       │                 │
│              │  ┌─────────────┐  │                 │
│              │  │ Embed query  │  │ SentenceBERT   │
│              │  └──────┬──────┘  │                 │
│              │         ▼         │                 │
│              │  ┌─────────────┐  │                 │
│              │  │ ChromaDB    │  │ cosine search   │
│              │  │ search      │  │                 │
│              │  └──┬──────┬──┘  │                 │
│              │     │      │     │                 │
│              │  HIT │      │ MISS│                 │
│              │     ▼      ▼     │                 │
│              │ ┌──────┐ ┌────┐  │                 │
│              │ │Cached│ │LLM │  │                 │
│              │ │answer│ │Call│  │                 │
│              │ └──┬───┘ └─┬──┘  │                 │
│              │    │       │     │                 │
│              │    │   ┌───▼──┐  │                 │
│              │    │   │Store │  │  Q+A → ChromaDB │
│              │    │   │in DB │  │                 │
│              │    │   └─────┘  │                 │
│              └────┼────────────┘                 │
│                   ▼                               │
│            ┌──────────────┐                       │
│            │ Discord reply │  with 👍👎 reactions │
│            └──────────────┘                       │
└──────────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│              LLM BACKENDS (pluggable)              │
│  ┌──────────┐ ┌────────┐ ┌──────────┐ ┌───────┐  │
│  │Claude    │ │OpenAI  │ │Anthropic │ │Ollama │  │
│  │Code CLI  │ │SDK     │ │SDK       │ │REST   │  │
│  └──────────┘ └────────┘ └──────────┘ └───────┘  │
│        ↑ all inherit from BaseBackend                │
└──────────────────────────────────────────────────┘
```

## Key Design Decisions

### Why ChromaDB + SentenceBERT instead of OpenAI embeddings?
- **Zero API cost** for embedding: SentenceBERT runs locally (~80MB)
- **Offline capable**: works without internet if using Ollama
- **Privacy**: no data leaves your server for FAQ matching
- **Speed**: <5ms per embedding on CPU

### Why adaptive threshold?
Manual threshold tuning is fragile:
- Too low → wrong FAQs served (bad UX)
- Too high → cache rarely hits (defeats purpose)

Adaptive mode auto-tunes based on actual hit rate. If hits are low, it lowers the threshold. If hits are too aggressive (false positives), it raises it.

### Why pluggable backends instead of just OpenAI SDK?
- Different teams have different LLM preferences
- Claude Code CLI is zero-SDK for Anthropic users
- Ollama for fully local/offline deployments
- Custom backend for enterprise API gateways

### FAQ lifecycle

```
New question → LLM answers → stored in cache (count=1, rating=0)
    ↓
Same question asked again → cache hit → count bumped to 2
    ↓
User reacts 👎 → rating drops
User reacts 👍 → rating rises
    ↓
rating hits -3 → FAQ auto-deleted
count hits 100 → FAQ considered "verified" (good signal)
```

### Feedback threshold (hardcoded 0.80)
Separate from cache threshold. Must be high to prevent misattribution:
- If user clicks 👎 on a message answering "how to install", we MUST verify the reaction maps to the right FAQ
- 0.80 is conservative: only match if the question text is very similar to the original
