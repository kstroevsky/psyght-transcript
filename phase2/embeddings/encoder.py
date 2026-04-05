"""Shared multilingual embedding encoder used by transcript indexing and query search."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

_model: Any = None


def get_encoder() -> "SentenceTransformer":
    """Lazily load the multilingual-e5-large encoder so imports stay lightweight."""

    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("intfloat/multilingual-e5-large")
    return _model


def _encode(texts: list[str], prefix: str) -> list[list[float]]:
    """Apply the required e5 prefixing rule before normalization and batching."""

    prefixed = [f"{prefix}: {text}" for text in texts]
    vectors = get_encoder().encode(prefixed, normalize_embeddings=True, batch_size=32)
    return vectors.tolist()


def encode_passages(texts: list[str]) -> list[list[float]]:
    """Encode transcript segments for storage in the `segments.embedding` column."""

    return _encode(texts, "passage")


def encode_queries(texts: list[str]) -> list[list[float]]:
    """Encode search queries for cosine similarity against stored segment embeddings."""

    return _encode(texts, "query")
