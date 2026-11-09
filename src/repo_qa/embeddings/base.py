"""Embedding provider protocol and factory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from repo_qa.config import Settings
    from repo_qa.models import CodeChunk


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into L2-normalised float32 vectors.

    ``name`` identifies the model and is part of the embedding cache key, so two providers
    must never share a name unless they produce identical vectors.
    """

    name: str

    @property
    def dim(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Shape ``(len(texts), dim)``."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Shape ``(dim,)``."""
        ...


def embedding_input(path: str, chunk: CodeChunk) -> str:
    """What actually gets embedded: location and symbol give the model useful context."""
    header = f"{path} {chunk.symbol}" if chunk.symbol else path
    return f"{header}\n{chunk.text}"


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def get_embedder(settings: Settings) -> EmbeddingProvider:
    if settings.embedder == "mock":
        from repo_qa.embeddings.mock import MockEmbedder

        return MockEmbedder()
    from repo_qa.embeddings.fastembed_provider import FastEmbedProvider

    return FastEmbedProvider(settings.embed_model)
