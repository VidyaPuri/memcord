# Contributing to Memcord

## Setup

```bash
git clone https://github.com/VidyaPuri/memcord
cd memcord
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[openai,anthropic,ollama]"
```

## Running Tests

```bash
pytest tests/ -v
```

First run downloads SentenceBERT (~80MB) — subsequent runs are instant.

## Adding a New Backend

1. Implement `LLMBackend` protocol in `memcord/backends/your_backend.py`:

```python
from memcord.backends import LLMBackend

class MyBackend:
    async def ask(self, prompt: str, system: str | None = None) -> str:
        # Your implementation here
        return response
```

2. Register it in `memcord/backends/__init__.py` factory function

3. Add optional dependency in `pyproject.toml`

4. Add env var docs to `.env.example`

5. Test: `MEMCORD_BACKEND=custom MEMCORD_CUSTOM_BACKEND=mymod.MyBackend memcord run`

## Code Style

- Type hints on all public methods
- Google-style docstrings
- Async where possible (discord.py is fully async)
- `from __future__ import annotations` at top of every file

## Pull Requests

1. Fork → branch → PR
2. All tests must pass
3. New features need tests
4. Keep PRs focused (one feature/fix per PR)

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md)
