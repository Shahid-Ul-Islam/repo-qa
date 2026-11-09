"""Brute-force cosine similarity over an in-memory matrix of chunk vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from repo_qa.store.repository import Store


@dataclass
class VectorIndex:
    chunk_ids: np.ndarray  # (n,) int64
    matrix: np.ndarray  # (n, dim) float32, rows L2-normalised

    @classmethod
    def load(cls, store: Store, repo_id: int, model: str) -> VectorIndex:
        ids, matrix = store.load_vectors(repo_id, model)
        return cls(ids, matrix)

    def __len__(self) -> int:
        return int(self.chunk_ids.shape[0])

    def search(
        self, query_vec: np.ndarray, limit: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Top ``limit`` ``(chunk_id, cosine)`` pairs, optionally restricted to ``allowed``."""
        if len(self) == 0 or limit <= 0:
            return []
        if query_vec.shape[0] != self.matrix.shape[1]:
            raise ValueError(
                f"query has dim {query_vec.shape[0]} but index has dim {self.matrix.shape[1]}"
            )
        scores = self.matrix @ query_vec.astype(np.float32)
        if allowed is not None:
            mask = np.isin(self.chunk_ids, np.fromiter(allowed, dtype=np.int64))
            scores = np.where(mask, scores, -np.inf)
        k = min(limit, len(self))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(self.chunk_ids[i]), float(scores[i])) for i in top if np.isfinite(scores[i])]
