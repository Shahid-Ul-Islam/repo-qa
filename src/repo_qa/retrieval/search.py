"""Hybrid search facade: vector + keyword, fused with reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence

from repo_qa.embeddings.base import EmbeddingProvider
from repo_qa.models import RepoRecord, SearchHit
from repo_qa.retrieval.fusion import RRF_K, PathFilter, rrf
from repo_qa.retrieval.keyword import keyword_search
from repo_qa.retrieval.vector import VectorIndex
from repo_qa.store.repository import Store


class IndexMismatchError(RuntimeError):
    """The repo was embedded with a different model than the configured embedder."""


class Searcher:
    def __init__(
        self,
        store: Store,
        embedder: EmbeddingProvider,
        *,
        candidates: int = 50,
        rrf_k: int = RRF_K,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.candidates = candidates
        self.rrf_k = rrf_k
        self._cache: dict[tuple[int, str, str], VectorIndex] = {}

    def _vectors(self, repo: RepoRecord) -> VectorIndex:
        key = (repo.id, self.embedder.name, str(repo.indexed_at))
        if key not in self._cache:
            self._cache = {k: v for k, v in self._cache.items() if k[0] != repo.id}
            self._cache[key] = VectorIndex.load(self.store, repo.id, self.embedder.name)
        return self._cache[key]

    def search(
        self,
        repo: RepoRecord,
        query: str,
        *,
        top_k: int = 8,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> list[SearchHit]:
        if repo.embed_model and repo.embed_model != self.embedder.name:
            raise IndexMismatchError(
                f"'{repo.name}' was indexed with embedding model '{repo.embed_model}' but the "
                f"configured embedder is '{self.embedder.name}'. Re-run `repo-qa index` or "
                "change REPO_QA_EMBEDDER / REPO_QA_EMBED_MODEL."
            )
        path_filter = PathFilter(include, exclude)
        allowed: set[int] | None = None
        if path_filter.active:
            allowed = {
                cid for cid, path in self.store.chunk_paths(repo.id).items() if path_filter(path)
            }
            if not allowed:
                return []

        vec_hits = self._vectors(repo).search(
            self.embedder.embed_query(query), self.candidates, allowed
        )
        # FTS cannot take the allow-list, so over-fetch and filter afterwards.
        kw_limit = self.candidates * (4 if allowed is not None else 1)
        kw_hits = keyword_search(self.store, repo.id, query, kw_limit)
        if allowed is not None:
            kw_hits = [h for h in kw_hits if h[0] in allowed][: self.candidates]

        fused = rrf([[cid for cid, _ in vec_hits], [cid for cid, _ in kw_hits]], k=self.rrf_k)
        ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        chunks = self.store.get_chunks([cid for cid, _ in ranked])
        vec_info = {cid: (rank, s) for rank, (cid, s) in enumerate(vec_hits, start=1)}
        kw_info = {cid: (rank, s) for rank, (cid, s) in enumerate(kw_hits, start=1)}

        hits: list[SearchHit] = []
        for cid, score in ranked:
            if cid not in chunks:
                continue
            v_rank, v_score = vec_info.get(cid, (None, None))
            k_rank, k_score = kw_info.get(cid, (None, None))
            hits.append(
                SearchHit(
                    chunk=chunks[cid],
                    score=score,
                    vector_score=v_score,
                    vector_rank=v_rank,
                    keyword_score=k_score,
                    keyword_rank=k_rank,
                )
            )
        return hits
