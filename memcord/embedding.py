"""Config-driven embedding model loading.

Provides a default SentenceTransformer model and a factory for building
custom embedding callables.  Used by SemanticAnswerCache.
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

    - None → default SentenceTransformer (all-MiniLM-L6-v2)
    - str  → SentenceTransformer(model_name)
    - EmbeddingModel → returned as-is
    """
    if embed_model is None:
        return get_default_model()
    if isinstance(embed_model, str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(embed_model)  # type: ignore[return-value]
    return embed_model
