"""Wires settings into concrete components; shared by the CLI and the web app."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from repo_qa.answering.providers.base import LLMProvider, get_provider
from repo_qa.answering.qa import QAEngine
from repo_qa.config import Settings
from repo_qa.embeddings.base import EmbeddingProvider, get_embedder
from repo_qa.indexing.indexer import Indexer
from repo_qa.models import RepoRecord
from repo_qa.retrieval.search import Searcher
from repo_qa.store.repository import Store


class RepoNotFoundError(LookupError):
    pass


@dataclass
class Runtime:
    settings: Settings
    store: Store
    embedder: EmbeddingProvider
    provider: LLMProvider
    searcher: Searcher = field(init=False)

    def __post_init__(self) -> None:
        self.searcher = Searcher(self.store, self.embedder)

    @classmethod
    def create(
        cls,
        settings: Settings,
        *,
        embedder: EmbeddingProvider | None = None,
        provider: LLMProvider | None = None,
    ) -> Runtime:
        return cls(
            settings=settings,
            store=Store.open(settings.db_path),
            embedder=embedder or get_embedder(settings),
            provider=provider or get_provider(settings),
        )

    def close(self) -> None:
        self.store.close()

    def indexer(self) -> Indexer:
        return Indexer(self.store, self.embedder, max_file_bytes=self.settings.max_file_bytes)

    def engine(self, *, top_k: int | None = None) -> QAEngine:
        s = self.settings
        return QAEngine(
            self.store,
            self.searcher,
            self.provider,
            top_k=top_k or s.top_k,
            token_budget=s.token_budget,
            min_vector_score=s.min_vector_score,
            history_turns=s.history_turns,
        )

    def resolve_repo(self, ref: str | None = None, cwd: Path | None = None) -> RepoRecord:
        """Find an indexed repo by id, name or path; with no ref, use the one containing
        ``cwd`` or the only indexed repo."""
        repos = self.store.list_repos()
        if not repos:
            raise RepoNotFoundError("No repositories indexed yet. Run `repo-qa index <path>`.")
        if ref:
            if ref.isdigit() and (repo := self.store.get_repo(int(ref))):
                return repo
            by_path = self.store.get_repo_by_path(str(Path(ref).expanduser().resolve()))
            if by_path:
                return by_path
            named = [r for r in repos if r.name == ref]
            if len(named) == 1:
                return named[0]
            raise RepoNotFoundError(f"No indexed repo matches '{ref}'. {self._choices(repos)}")
        here = (cwd or Path.cwd()).resolve()
        containing = [r for r in repos if here.is_relative_to(Path(r.root_path))]
        if containing:
            return max(containing, key=lambda r: len(r.root_path))
        if len(repos) == 1:
            return repos[0]
        raise RepoNotFoundError(
            f"Several repos are indexed; pick one with --repo. {self._choices(repos)}"
        )

    @staticmethod
    def _choices(repos: list[RepoRecord]) -> str:
        return "Indexed: " + ", ".join(f"{r.id}={r.name} ({r.root_path})" for r in repos)
