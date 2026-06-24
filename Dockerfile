# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv

WORKDIR /build

# Copy packaging metadata first for layer caching
COPY pyproject.toml ./

# Create venv and install core runtime deps
RUN uv venv /opt/venv && \
    uv pip install --python /opt/venv/bin/python \
        "discord.py>=2.3" \
        "chromadb>=0.4" \
        "sentence-transformers>=2.2" \
        "python-dotenv>=1.0" \
        "aiohttp>=3.9"

# ── Stage 2: Runner ───────────────────────────────────────
FROM python:3.12-slim AS runner

RUN groupadd -r memcord && useradd -r -g memcord -d /app memcord

# Copy virtualenv
COPY --from=builder --chown=memcord:memcord /opt/venv /opt/venv

# Copy application code
WORKDIR /app
COPY --chown=memcord:memcord . .

# Set environment — uses python -m entrypoint, no pip install needed
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MEMCORD_DATA_DIR=/app/memcord_data \
    MEMCORD_HEALTH_PORT=8080

# Volume for persistent FAQ cache data
VOLUME ["/app/memcord_data"]

USER memcord

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

ENTRYPOINT ["python3", "-m", "memcord.cli"]
CMD ["run"]
