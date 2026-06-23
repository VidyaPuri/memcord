"""Security hardening — input sanitization, prompt injection detection, env validation."""

from __future__ import annotations

import logging
import os
import re
import string
import sys

log = logging.getLogger("memcord.security")

# ── control characters to strip ─────────────────────────────
# Allow printable ASCII, common whitespace, and Unicode letters
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# Excessive whitespace: 3+ newlines, 3+ spaces
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE = re.compile(r" {3,}")


def sanitize_input(text: str) -> str:
    """Strip control chars, null bytes, and normalize excessive whitespace.

    Returns the sanitized string.
    """
    if not text:
        return ""

    # 1. Strip null bytes and control characters
    text = _CONTROL_CHARS.sub("", text)

    # 2. Collapse excessive newlines (3+ → 2)
    text = _MULTI_NEWLINE.sub("\n\n", text)

    # 3. Collapse excessive spaces (3+ → 1) — but preserve indentation intent
    text = _MULTI_SPACE.sub(" ", text)

    # 4. Strip leading/trailing whitespace
    text = text.strip()

    return text


# ── prompt injection patterns ───────────────────────────────

# Common prompt injection phrases — all lowercase for case-insensitive matching
_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|messages?)",
    r"forget\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|messages?)",
    r"disregard\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|messages?)",
    r"override\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?|messages?)",
    r"you\s+are\s+now\s+(a\s+)?(dan|jailbroken|unfiltered|unrestricted)",
    r"system\s*prompt\s*[:=]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"<\/?system>",
    r"\[system\]",
    r"\.\.\.\s*Now\s+(you\s+)?(are|must)",
    r"pretend\s+(you\s+are|to\s+be)",
    r"act\s+as\s+(if\s+)?(you\s+are|a\s+different)",
    r"what\s+is\s+your\s+(system\s+)?prompt",
    r"reveal\s+your\s+(system\s+)?(prompt|instructions)",
    r"print\s+your\s+(system\s+)?(prompt|instructions)",
    r"show\s+(me\s+)?your\s+(system\s+)?(prompt|instructions)",
    r"tell\s+(me\s+)?your\s+(system\s+)?(prompt|instructions)",
    r"output\s+your\s+(system\s+)?(prompt|instructions)",
    # Token / API key extraction attempts
    r"(api|discord|bot)\s*(key|token|secret)\s*[:=]",
    r"what\s+is\s+(the|your)\s+(api|discord|bot)\s*(key|token|secret)",
    # Delimiter injection
    r"####{3,}",
    r"-----{3,}",
    r"```{3,}",
    # Role-playing as system
    r"^(system|assistant|user)\s*:\s*",
]

_INJECTION_RE = re.compile(
    "|".join(f"({p})" for p in _INJECTION_PATTERNS),
    re.IGNORECASE,
)


def check_prompt_injection(text: str) -> bool:
    """Check if *text* contains prompt injection patterns.

    Returns True if potential injection detected.
    """
    if not text or not text.strip():
        return False
    return bool(_INJECTION_RE.search(text))


# ── env validation ──────────────────────────────────────────

# Known placeholder patterns
_PLACEHOLDER_PATTERNS = [
    r"^\**$",                     # all asterisks
    r"^<.*>$",                    # angle-bracket placeholder
    r"^your[-_].*$",              # "your-token-here"
    r"^change[-_]?me$",           # "changeme"
    r"^xxx+$",                    # "xxx"
    r"^placeholder$",
    r"^put[-_].*here$",
]


def _is_placeholder(value: str) -> bool:
    """Check if a value looks like a placeholder rather than a real value."""
    if not value:
        return True
    return any(re.match(p, value, re.IGNORECASE) for p in _PLACEHOLDER_PATTERNS)


def validate_env() -> None:
    """Validate required environment variables. Exits with clear message on failure.

    Checks:
    - DISCORD_TOKEN is set and not a placeholder
    - Backend-specific API keys are present when needed
    - Numeric env vars have sensible values
    """
    errors: list[str] = []

    # ── DISCORD_TOKEN ──
    token = os.getenv("DISCORD_TOKEN", "")
    if not token:
        errors.append("DISCORD_TOKEN is not set in .env — this is required.")
    elif _is_placeholder(token):
        errors.append(
            "DISCORD_TOKEN appears to be a placeholder (e.g. '***', '<your-token>'). "
            "Set it to your actual Discord bot token."
        )

    # ── Backend-specific checks ──
    backend = os.getenv("MEMCORD_BACKEND", "claude_code").lower()
    if backend == "claude_code":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            errors.append(
                "MEMCORD_BACKEND=claude_code requires ANTHROPIC_API_KEY to be set."
            )
    elif backend == "openai":
        openai_key = os.getenv("OPENAI_API_KEY", "")
        if not openai_key:
            errors.append("MEMCORD_BACKEND=openai requires OPENAI_API_KEY to be set.")
    elif backend == "anthropic":
        anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not anthropic_key:
            errors.append(
                "MEMCORD_BACKEND=anthropic requires ANTHROPIC_API_KEY to be set."
            )

    # ── Threshold validation ──
    sim_threshold = os.getenv("MEMCORD_SIMILARITY_THRESHOLD", "0.80")
    try:
        val = float(sim_threshold)
        if not 0.0 <= val <= 1.0:
            errors.append(
                f"MEMCORD_SIMILARITY_THRESHOLD={sim_threshold!r} is out of range (0.0–1.0)."
            )
    except ValueError:
        errors.append(
            f"MEMCORD_SIMILARITY_THRESHOLD={sim_threshold!r} is not a valid float."
        )

    feedback_threshold = os.getenv("MEMCORD_FEEDBACK_THRESHOLD", "0.80")
    try:
        val = float(feedback_threshold)
        if not 0.0 <= val <= 1.0:
            errors.append(
                f"MEMCORD_FEEDBACK_THRESHOLD={feedback_threshold!r} is out of range (0.0–1.0)."
            )
    except ValueError:
        errors.append(
            f"MEMCORD_FEEDBACK_THRESHOLD={feedback_threshold!r} is not a valid float."
        )

    # ── Health port ──
    health_port = os.getenv("MEMCORD_HEALTH_PORT", "8080")
    try:
        val = int(health_port)
        if val < 0 or val > 65535:
            errors.append(
                f"MEMCORD_HEALTH_PORT={health_port!r} is out of range (0–65535)."
            )
    except ValueError:
        errors.append(f"MEMCORD_HEALTH_PORT={health_port!r} is not a valid integer.")

    # ── Max input length ──
    max_input = os.getenv("MEMCORD_MAX_INPUT_LENGTH", "2000")
    try:
        val = int(max_input)
        if val < 1:
            errors.append(
                f"MEMCORD_MAX_INPUT_LENGTH={max_input!r} must be a positive integer."
            )
    except ValueError:
        errors.append(
            f"MEMCORD_MAX_INPUT_LENGTH={max_input!r} is not a valid integer."
        )

    # ── Report errors ──
    if errors:
        log.error("Environment validation failed:")
        for err in errors:
            log.error(f"  • {err}")
        sys.exit(1)
