"""Model presets for different languages and quality/speed trade-offs.

Usage:
    MEMCORD_EMBEDDING_MODEL=slovenian   # uses LaBSE
    MEMCORD_EMBEDDING_MODEL=fast        # uses all-MiniLM-L6-v2 (default)
    MEMCORD_EMBEDDING_MODEL=accurate    # uses all-mpnet-base-v2
"""

from __future__ import annotations

# Map of preset names to HuggingFace model identifiers.
# Add your own presets here.
PRESETS: dict[str, str] = {
    # ── English ──
    "fast": "all-MiniLM-L6-v2",       # 80 MB, good enough for FAQ
    "accurate": "all-mpnet-base-v2",   # 420 MB, best English quality
    "tiny": "paraphrase-MiniLM-L3-v2", # 60 MB, ultra-fast

    # ── Multilingual ──
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",  # 470 MB
    "multilingual-tiny": "paraphrase-multilingual-MiniLM-L3-v2",  # 190 MB

    # ── Slovenian ──
    "slovenian": "sentence-transformers/LaBSE",  # 1.8 GB, 109 languages, ★★★★★ Slovenian
    "slovenian-fast": "paraphrase-multilingual-MiniLM-L12-v2",  # ★★★☆☆ Slovenian

    # ── Other languages ──
    "labse": "sentence-transformers/LaBSE",  # 109 languages, best for non-English
}


def resolve_model(name: str) -> str:
    """Resolve a preset name or raw HuggingFace model ID.

    Returns the HuggingFace model string. Pass-through if not a known preset.
    """
    return PRESETS.get(name, name)
