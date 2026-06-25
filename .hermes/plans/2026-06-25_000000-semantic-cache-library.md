# Semantic Cache Library — Implementation Plan

> **For Hermes:** Use test-driven-development skill to implement task-by-task. RED → GREEN → REFACTOR per task.

**Goal:** Extract `FAQCache` into a reusable `SemanticAnswerCache` library with scope partitioning, caller-driven hooks, conservative promotion mode, and pluggable embeddings — while keeping the bundled Discord bot running unchanged.

**Architecture:** `SemanticAnswerCache` becomes the core implementation in `memcord/cache/semantic_cache.py`. `FAQCache` becomes a thin backward-compatible subclass exposing the old API (`check`/`store`/`feedback`) as wrappers around the new API (`lookup`/`observe`/`vote`). A `build_cache()` factory function sits at module level. Scope is stored in ChromaDB metadata and used as a `where` filter. `promote_after` uses an `observation_count` field — entries are invisible to `lookup` until observed N times.

**Tech Stack:** Python 3.10+, ChromaDB, SentenceTransformers, pytest

---

## Interface Contract (shared between memcord and consumers)

### Canonical provider option: `data_dir`

The only **required** provider option passed through `build_cache(**provider_options)`
is ``data_dir`` (str) — the path to the cache data directory.

**Consumer teams MUST pass `data_dir=...`, not `cache_dir`.**

| Is | Isn't |
|----|-------|
| `build_cache(data_dir="/path")` ✅ | `build_cache(cache_dir="/path")` ❌  — TypeError |

The parameter maps directly to `SemanticAnswerCache.__init__(data_dir=...)`.
Passing a different name causes `TypeError: unexpected keyword argument` and the
cache silently fails to build.

### Common optional provider options

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| `embed_model` | str \| object \| None | `None` (→ `all-MiniLM-L6-v2`) | Model name or callable with `.encode()` |
| `similarity_threshold` | float | `0.80` | Cosine similarity floor for a hit |
| `adaptive_threshold` | bool | `True` | Auto-tune threshold based on hit rate |
| `consolidate_threshold` | float | `0.95` | Similarity above which `observe` merges into an existing entry |
| `feedback_threshold` | float | `0.80` (env: `MEMCORD_FEEDBACK_THRESHOLD`) | Minimum similarity for feedback attribution |

### Promotion threshold pitfall

`observe()` merges a new observation into an existing entry only when similarity ≥
`consolidate_threshold` (default 0.95). `lookup()` matches at `similarity_threshold`
(default 0.80). Paraphrases in the 0.80–0.95 band each spawn separate singletons
that never accumulate toward `promote_after` and remain invisible.

If your use case expects varied phrasings to promote, lower `consolidate_threshold`
closer to `similarity_threshold`:

```python
cache = build_cache(data_dir="./cache", promote_after=3,
                    similarity_threshold=0.80, consolidate_threshold=0.82)
```

---

## Key Design Decisions

### Scope filtering via ChromaDB `where` clause
ChromaDB's `query()` supports `where={"scope": scope}` for exact-match metadata filtering. Since scope is an opaque partition key, exact match is exactly right.

### `promote_after` via `observation_count` metadata
Each entry gets an `observation_count` field. `observe()` increments it. `lookup()` only returns entries where `observation_count >= promote_after`. This means:
- `promote_after=1`: immediate promotion (current behavior)
- `promote_after=5`: entry must be observed 5 times before it's ever returned

### Schema migration (v1 → v2)
Existing ChromaDB entries lack `scope` and `observation_count`. Migration:
1. Add `scope = scope_default` to all existing entries
2. Set `observation_count = max(existing_count, 100)` so old entries are immediately promoted
3. Bump `SCHEMA_VERSION` to 2

### `FAQCache` as backward-compatible subclass
`FAQCache(SemanticAnswerCache)` inherits all new functionality and exposes the old surface:
- `check(q)` → `lookup(q, scope=self.scope_default)` returning `(answer, similarity)` tuple
- `store(q, a)` → `observe(q, a, scope=self.scope_default)`
- `feedback(q, positive)` → searches for the FAQ, then calls `vote(id, +/-1)`
- All old properties preserved: `stats`, `threshold`, `adaptive`, `list_faqs`, `consolidate`, `reset`, `flush_stats`, `set_metrics`

### No Discord/CLI imports in cache module
The cache module (`memcord/cache/`) must not import from `memcord.discord_`, `memcord.cli`, or `memcord.backends`. Currently it already doesn't — we just verify this stays true.

---

## Tasks

### Task 1: Define `CacheHit` dataclass and `AnswerCache` Protocol

**Objective:** Add the new types to `memcord/cache/__init__.py`

**Files:**
- Modify: `memcord/cache/__init__.py`

**Step 1: Write the types**

```python
"""Semantic FAQ cache — public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memcord.cache.faq_cache import FAQCache
from memcord.cache.semantic_cache import SemanticAnswerCache, build_cache


@dataclass
class CacheHit:
    """A cache hit — returned by lookup()."""

    id: str
    answer: str


class AnswerCache(Protocol):
    """Protocol for any cache implementing the semantic-cache interface."""

    def lookup(self, question: str, *, scope: str) -> CacheHit | None: ...
    def observe(self, question: str, answer: str, *, scope: str) -> None: ...
    def vote(self, hit_id: str, delta: int) -> None: ...


__all__ = [
    "FAQCache",
    "SemanticAnswerCache",
    "build_cache",
    "CacheHit",
    "AnswerCache",
]
```

Note: `SemanticAnswerCache` and `build_cache` don't exist yet — this import will fail. That's fine; we're building in order. For now, just add the `CacheHit` dataclass and `AnswerCache` protocol. The imports for `SemanticAnswerCache` and `build_cache` get added in Task 3.

**Verification:** `from memcord.cache import CacheHit, AnswerCache` works

**Step 2: Commit**

```bash
git add memcord/cache/__init__.py
git commit -m "feat: add CacheHit dataclass and AnswerCache Protocol"
```

---

### Task 2: Create `SemanticAnswerCache` — core implementation

**Objective:** New file `memcord/cache/semantic_cache.py` with the full `SemanticAnswerCache` class and `build_cache()` factory

**Files:**
- Create: `memcord/cache/semantic_cache.py`

**Step 1: Write the file**

The file contains:
1. `SemanticAnswerCache` class — port of `FAQCache` with:
   - New `__init__` accepting `scope_default`, `promote_after`, `should_cache`, `validate`, `embed_model`
   - `lookup(question, *, scope) -> CacheHit | None` (replaces `check`)
   - `observe(question, answer, *, scope) -> None` (replaces `store`)
   - `vote(hit_id, delta) -> None` (replaces `feedback`)
   - All existing functionality: adaptive threshold, consolidation, stats, batched writes, lazy model loading, feedback-threshold gating, downvote-prune
   - Scope filtering via ChromaDB `where` clause
   - `promote_after` gating via `observation_count` metadata
   - `should_cache` gating in `observe`
   - `validate` gating in `lookup`
   - `embed_model` parameter — if callable, use directly; if string, pass to SentenceTransformer; if None, default to `"all-MiniLM-L6-v2"`
2. `build_cache()` factory function

**Step 2: Run existing tests to ensure no breakage**

```bash
pytest tests/test_cache.py -v
```

(These test `FAQCache` directly — will fail if `FAQCache` hasn't been updated yet. That's Task 4.)

**Step 3: Commit**

```bash
git add memcord/cache/semantic_cache.py
git commit -m "feat: add SemanticAnswerCache with scope, hooks, promote_after"
```

---

### Task 3: Update `memcord/cache/__init__.py` exports

**Objective:** Add `SemanticAnswerCache` and `build_cache` imports

**Files:**
- Modify: `memcord/cache/__init__.py`

**Step 1: Update imports** (merge with Task 1 changes)

```python
"""Semantic FAQ cache — public API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from memcord.cache.faq_cache import FAQCache
from memcord.cache.semantic_cache import SemanticAnswerCache, build_cache


@dataclass
class CacheHit:
    """A cache hit — returned by lookup()."""

    id: str
    answer: str


class AnswerCache(Protocol):
    """Protocol for any cache implementing the semantic-cache interface."""

    def lookup(self, question: str, *, scope: str) -> CacheHit | None: ...
    def observe(self, question: str, answer: str, *, scope: str) -> None: ...
    def vote(self, hit_id: str, delta: int) -> None: ...


__all__ = [
    "FAQCache",
    "SemanticAnswerCache",
    "build_cache",
    "CacheHit",
    "AnswerCache",
]
```

**Step 2: Verify import works**

```bash
python -c "from memcord.cache import SemanticAnswerCache, build_cache, CacheHit, AnswerCache; print('OK')"
```

**Step 3: Commit**

```bash
git add memcord/cache/__init__.py
git commit -m "feat: export SemanticAnswerCache and build_cache"
```

---

### Task 4: Refactor `FAQCache` as backward-compatible subclass

**Objective:** `FAQCache` becomes `FAQCache(SemanticAnswerCache)` with old API preserved

**Files:**
- Modify: `memcord/cache/faq_cache.py`

**Step 1: Rewrite `FAQCache` as thin subclass**

The `FAQCache` class should:
- Inherit from `SemanticAnswerCache`
- Override `__init__` to accept old parameters and pass them through, plus `scope_default`, `promote_after=1`, etc.
- Add `check(question) -> tuple[str|None, float]` wrapper
- Add `store(question, answer) -> None` wrapper
- Add `feedback(question, positive) -> None` wrapper
- Preserve all old properties: `stats`, `threshold`, `adaptive`, `list_faqs`, `consolidate`, `reset`, `flush_stats`, `set_metrics`
- Keep `SCHEMA_VERSION` and migration logic

The body of `FAQCache` becomes very thin since all logic lives in `SemanticAnswerCache`.

**Step 2: Run existing cache tests**

```bash
pytest tests/test_cache.py -v
```

Expected: all tests pass (or adjust for minor API differences)

**Step 3: Run bot tests**

```bash
pytest tests/test_bot.py -v
```

Expected: all tests pass

**Step 4: Commit**

```bash
git add memcord/cache/faq_cache.py
git commit -m "refactor: FAQCache as backward-compatible subclass of SemanticAnswerCache"
```

---

### Task 5: Update `memcord/__init__.py` exports

**Objective:** Export the new surface from the package root

**Files:**
- Modify: `memcord/__init__.py`

**Step 1: Add exports**

```python
"""Memcord — LLM-agnostic Discord FAQ bot with semantic caching."""

from importlib.metadata import PackageNotFoundError, version

from memcord.cache import AnswerCache, CacheHit, FAQCache, SemanticAnswerCache, build_cache

try:
    __version__ = version("memcord")
except PackageNotFoundError:
    __version__ = "0.1.0+local"

__all__ = [
    "__version__",
    "FAQCache",
    "SemanticAnswerCache",
    "build_cache",
    "CacheHit",
    "AnswerCache",
]
```

**Step 2: Verify import**

```bash
python -c "from memcord import SemanticAnswerCache, build_cache, CacheHit, AnswerCache; print('OK')"
```

**Step 3: Verify no Discord/CLI deps pulled in**

```bash
python -c "import memcord.cache; print('No discord/CLI imports triggered')"
```

**Step 4: Commit**

```bash
git add memcord/__init__.py
git commit -m "feat: export SemanticAnswerCache, build_cache, CacheHit from memcord root"
```

---

### Task 6: Implement `memcord/embedding.py` — config-driven embedding model

**Objective:** Make `embedding.py` a reusable module for embedding model loading

**Files:**
- Modify: `memcord/embedding.py`

**Step 1: Write the module**

```python
"""Config-driven embedding model loading.

Provides a default SentenceTransformer model and a factory for building
custom embedding callables. Used by SemanticAnswerCache.
"""

from __future__ import annotations

from typing import Protocol


class EmbeddingModel(Protocol):
    """Protocol for embedding models — callable that takes list[str] → list[list[float]]."""

    def encode(self, texts: list[str]) -> list[list[float]]: ...


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_default_model() -> EmbeddingModel:
    """Return the default embedding model (lazy-loaded SentenceTransformer)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DEFAULT_MODEL_NAME)  # type: ignore[return-value]


def resolve_embed_model(embed_model: str | EmbeddingModel | None = None) -> EmbeddingModel:
    """Resolve an embedding model specification to an EmbeddingModel.

    - None → default SentenceTransformer
    - str → SentenceTransformer(model_name)
    - EmbeddingModel → returned as-is
    """
    if embed_model is None:
        return get_default_model()
    if isinstance(embed_model, str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(embed_model)  # type: ignore[return-value]
    return embed_model
```

**Step 2: Verify import**

```bash
python -c "from memcord.embedding import resolve_embed_model, DEFAULT_MODEL_NAME; print('OK')"
```

**Step 3: Commit**

```bash
git add memcord/embedding.py
git commit -m "feat: config-driven embedding model with resolve_embed_model"
```

---

### Task 7: New unit tests — scope isolation, should_cache, validate, promote_after

**Objective:** Add comprehensive tests for the new features

**Files:**
- Create: `tests/test_semantic_cache.py`

**Step 1: Write the test file** (TDD — write tests first)

Tests covering:
- `TestScopeIsolation`: scope="a" entry never returned for scope="b", same scope works, different scopes are independent
- `TestShouldCache`: `should_cache` returning False drops the answer, default lambda caches everything
- `TestValidate`: `validate` returning False causes lookup miss, default always validates, validate receives CacheHit
- `TestPromoteAfter`: `promote_after=3` means entry invisible until observed 3 times, `promote_after=1` immediate, scoped promote_after
- `TestBuildCache`: factory creates SemanticAnswerCache, passes options through
- `TestVoteMethod`: vote adjusts rating, vote downvote prune still works, vote on nonexistent id is no-op

**Step 2: Run new tests (expected: FAIL — SemanticAnswerCache exists but tests may need adjustment)**

```bash
pytest tests/test_semantic_cache.py -v
```

**Step 3: Fix any failures, then verify all pass**

```bash
pytest tests/test_semantic_cache.py tests/test_cache.py tests/test_bot.py -v
```

**Step 4: Commit**

```bash
git add tests/test_semantic_cache.py
git commit -m "test: add tests for scope, should_cache, validate, promote_after"
```

---

### Task 8: Update documentation

**Objective:** Reflect new surface in README.md and ARCHITECTURE.md

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`

**Step 1: Update README.md**

Add a "Semantic Cache Library" section with:
- Import example: `from memcord import SemanticAnswerCache, build_cache, CacheHit`
- Quick usage: `build_cache()`, `lookup()`, `observe()`, `vote()`
- Description of scope, should_cache, validate, promote_after
- Note that the bundled bot is one consumer

**Step 2: Update ARCHITECTURE.md**

Add a "Cache Library" section documenting:
- `SemanticAnswerCache` as the reusable core
- `FAQCache` as backward-compatible wrapper
- Scope partitioning, hooks, promotion gating
- Embedding model pluggability

**Step 3: Commit**

```bash
git add README.md ARCHITECTURE.md
git commit -m "docs: document SemanticAnswerCache library surface"
```

---

### Task 9: Full test suite + smoke test

**Objective:** Run everything and verify no regressions

**Step 1: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass

**Step 2: Verify clean import (no Discord/CLI deps)**

```bash
python -c "
import sys
before = set(sys.modules.keys())
from memcord.cache import SemanticAnswerCache, build_cache
after = set(sys.modules.keys())
new = after - before
bad = {m for m in new if 'discord' in m.lower() or 'cli' in m.lower()}
assert not bad, f'Unwanted imports: {bad}'
print('Clean import — no Discord/CLI deps')
"
```

**Step 3: Verify build_cache factory works**

```bash
python -c "
from memcord import build_cache, CacheHit
cache = build_cache(data_dir='/tmp/test_build_cache', promote_after=2)
print(f'Created: {type(cache).__name__}')
import shutil; shutil.rmtree('/tmp/test_build_cache', ignore_errors=True)
"
```

**Step 4: Commit if any fixes needed, otherwise done**

---

## Summary of Changes

| File | Action | Description |
|------|--------|-------------|
| `memcord/cache/__init__.py` | Modify | Add `CacheHit`, `AnswerCache`, `SemanticAnswerCache`, `build_cache` |
| `memcord/cache/semantic_cache.py` | **Create** | Core `SemanticAnswerCache` + `build_cache()` factory |
| `memcord/cache/faq_cache.py` | Modify | `FAQCache` → thin backward-compatible subclass |
| `memcord/__init__.py` | Modify | Export `SemanticAnswerCache`, `build_cache`, `CacheHit`, `AnswerCache` |
| `memcord/embedding.py` | Modify | Add `resolve_embed_model()`, `DEFAULT_MODEL_NAME`, `EmbeddingModel` |
| `tests/test_semantic_cache.py` | **Create** | Tests for scope, hooks, promote_after |
| `README.md` | Modify | Document reusable library surface |
| `ARCHITECTURE.md` | Modify | Document cache library architecture |

## Verification Checklist

- [ ] `from memcord import SemanticAnswerCache, build_cache, CacheHit` works
- [ ] `from memcord.cache import SemanticAnswerCache` imports with no Discord/CLI deps
- [ ] All existing tests pass (`test_cache.py`, `test_bot.py`)
- [ ] New tests pass (`test_semantic_cache.py`)
- [ ] Scope isolation: entry under scope="a" never returned for scope="b"
- [ ] `should_cache` gating: False → entry dropped
- [ ] `validate` gating: False → lookup miss
- [ ] `promote_after=3`: entry invisible until 3 observations
- [ ] `promote_after=1`: immediate promotion (current behavior)
- [ ] `vote(hit_id, -1)` adjusts rating, downvote-prune still works
- [ ] `build_cache()` factory produces a working `SemanticAnswerCache`
- [ ] Pluggable embedding model works
- [ ] Existing count/rating/adaptive-threshold/consolidation behavior preserved
- [ ] `FAQCache` backward-compatible API unchanged for bot
