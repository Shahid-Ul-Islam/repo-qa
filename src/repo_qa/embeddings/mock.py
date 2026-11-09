"""Deterministic, dependency-free embedder for tests and offline CI.

Feature-hashes word tokens (with compound identifiers split into parts) into a fixed-size
vector with sublinear term weighting. It has no semantics beyond lexical overlap, but that
is enough for retrieval tests to be meaningful and fully reproducible.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Sequence

import numpy as np

from repo_qa.embeddings.base import normalize_rows
from repo_qa.text import tokenize

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "does",
        "do",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "this",
        "that",
        "to",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
        "self",
        "return",
        "def",
        "import",
        "none",
        "true",
        "false",
    ]
)


class MockEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.name = f"mock-hash-{dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        counts = Counter(t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1)
        for token, tf in counts.items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self._dim
            sign = 1.0 if (value >> 63) & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(tf))
        return vec

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return normalize_rows(np.vstack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return normalize_rows(self._vector(text)[None, :])[0]
