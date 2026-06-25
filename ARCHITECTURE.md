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

## Cache Library Architecture

The FAQ cache is now a reusable library. `SemanticAnswerCache` (in `memcord/cache/semantic_cache.py`) is the core implementation — it has no knowledge of Discord, backends, or the CLI. `FAQCache` is a thin backward-compatible subclass that exposes the original API for the bundled bot.

```
┌────────────────────────────────────────────┐
│         SemanticAnswerCache (core)          │
│                                             │
│  lookup(q, scope)      → CacheHit | None   │
│  observe(q, a, scope)  → None              │
│  vote(hit_id, delta)   → None              │
│                                             │
│  Hooks:                                     │
│    should_cache(answer) → bool              │
│    validate(CacheHit)   → bool              │
│                                             │
│  Gates:                                     │
│    promote_after=N (observation_count)      │
│    scope partitioning (ChromaDB where)      │
│                                             │
│  Embedding: pluggable (embed_model=...)     │
└──────────────────┬─────────────────────────┘
                   │ inherits
┌──────────────────▼─────────────────────────┐
│              FAQCache (legacy)              │
│                                             │
│  check(q)      → (str|None, float)         │
│  store(q, a)   → None                      │
│  feedback(q, ±)→ None                      │
│                                             │
│  Used by: Discord bot (unchanged)           │
└────────────────────────────────────────────┘
```

### Scope Partitioning

Scope is an opaque string stored as ChromaDB metadata. All vector queries are filtered with `where={"scope": scope}`. An entry stored under `scope="support"` is never returned for `scope="sales"`.

### Promotion Gate (`promote_after`)

Each entry carries an `observation_count` metadata field. `observe()` increments it. `lookup()` only returns entries where `observation_count >= promote_after`. Setting `promote_after=1` reproduces the original immediate-promotion behavior.

### Hooks

- **`should_cache(answer)`**: Called in `observe()`. If it returns `False`, the answer is silently dropped. Default: `lambda _: True`.
- **`validate(CacheHit)`**: Called in `lookup()` on every candidate. If it returns `False`, the candidate is discarded. Default: `lambda _: True`.

### Pluggable Embedding

Pass `embed_model` to `build_cache()` or `SemanticAnswerCache.__init__()`:
- `None` → default `all-MiniLM-L6-v2` via SentenceTransformer
- `str` → model name for SentenceTransformer
- Any object with `.encode(list[str]) → list[list[float]]`

Resolution is handled by `memcord.embedding.resolve_embed_model()`, which is also
available for external use.

### CacheHit Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | ChromaDB document ID — usable with `vote()` |
| `answer` | `str` | The cached answer text |
| `similarity` | `float` | Cosine similarity score (0.0–1.0) of the match |

### Consolidation vs Lookup Threshold

`observe()` merges a new observation into an existing entry only when similarity ≥
`consolidate_threshold` (default 0.95). `lookup()` matches at `similarity_threshold`
(default 0.80). Paraphrases in the 0.80–0.95 band create separate singletons that
never accumulate toward `promote_after`. To treat varied phrasings as the same
question for promotion, lower `consolidate_threshold` closer to
`similarity_threshold`.

### Schema Migration

The collection auto-migrates from v1 → v2 on first access:
- Adds `scope` (defaulting to `scope_default`) to all existing entries
- Adds `observation_count` (set to a high value so existing entries are immediately promoted)

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
