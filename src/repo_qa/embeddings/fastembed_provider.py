"""Local ONNX embeddings via fastembed (no API key; model downloaded on first use)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from repo_qa.embeddings.base import normalize_rows

# Avoids importing fastembed (slow) just to learn the dimension of common models.
KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5": 768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}


class FastEmbedProvider:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64) -> None:
        self.name = model_name
        self.batch_size = batch_size
        self._model: Any = None
        self._dim: int | None = KNOWN_DIMS.get(model_name)

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self.name)
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            from fastembed import TextEmbedding

            for info in TextEmbedding.list_supported_models():
                if info["model"] == self.name:
                    self._dim = int(info["dim"])
                    break
            else:
                self._dim = int(self.embed_query("probe").shape[0])
        return self._dim

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = list(self._load().passage_embed(list(texts), batch_size=self.batch_size))
        return normalize_rows(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        vector = next(iter(self._load().query_embed([text])))
        return normalize_rows(np.asarray(vector)[None, :])[0]
