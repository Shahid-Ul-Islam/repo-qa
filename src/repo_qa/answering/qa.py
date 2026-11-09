"""Question answering: retrieve → gate on confidence → prompt → verify citations → remember."""

from __future__ import annotations

from collections.abc import Sequence

from repo_qa.answering.citations import check_answer
from repo_qa.answering.memory import SessionMemory, retrieval_query
from repo_qa.answering.prompts import build_request, is_not_found
from repo_qa.answering.providers.base import LLMProvider
from repo_qa.models import Answer, RepoRecord, SearchHit
from repo_qa.retrieval.context import build_context_pack
from repo_qa.retrieval.search import Searcher
from repo_qa.store.repository import Store

NOT_FOUND_MESSAGE = (
    "Not found in the repo: the indexed code does not appear to contain an answer to this question."
)

# Cosine similarity at which a vector-only match is trusted. Mock vectors are sparse
# hashed bags of words; fastembed (bge) similarities sit higher overall.
DEFAULT_MIN_VECTOR_SCORE = {"mock": 0.5, "default": 0.65}


def default_min_vector_score(embedder_name: str) -> float:
    if embedder_name.startswith("mock"):
        return DEFAULT_MIN_VECTOR_SCORE["mock"]
    return DEFAULT_MIN_VECTOR_SCORE["default"]


def retrieval_confidence(hits: Sequence[SearchHit], min_vector_score: float) -> str | None:
    """Why retrieval is trustworthy, or None when the answer should be "not found".

    Confident when the top hit matches query keywords (BM25 only counts non-stopword terms),
    or when some hit's cosine similarity clears ``min_vector_score``.
    """
    if not hits:
        return None
    if hits[0].keyword_rank is not None:
        return "keyword match"
    best = max((h.vector_score or 0.0) for h in hits)
    if best >= min_vector_score:
        return f"vector similarity {best:.2f}"
    return None


class QAEngine:
    def __init__(
        self,
        store: Store,
        searcher: Searcher,
        provider: LLMProvider,
        *,
        top_k: int = 8,
        token_budget: int = 6000,
        min_vector_score: float | None = None,
        history_turns: int = 4,
    ) -> None:
        self.store = store
        self.searcher = searcher
        self.provider = provider
        self.top_k = top_k
        self.token_budget = token_budget
        self.min_vector_score = (
            min_vector_score
            if min_vector_score is not None
            else default_min_vector_score(searcher.embedder.name)
        )
        self.memory = SessionMemory(store, history_turns)

    def ask(
        self,
        repo: RepoRecord,
        question: str,
        *,
        session_id: str | None = None,
        include: Sequence[str] = (),
        exclude: Sequence[str] = (),
    ) -> Answer:
        question = question.strip()
        if session_id and not self.memory.belongs_to(session_id, repo.id):
            session_id = None  # never mix conversations across repos
        history = self.memory.history(session_id)

        hits = self.searcher.search(
            repo,
            retrieval_query(question, history),
            top_k=self.top_k,
            include=include,
            exclude=exclude,
        )
        reason = retrieval_confidence(hits, self.min_vector_score)
        if reason is None:
            answer = Answer(
                question=question,
                text=NOT_FOUND_MESSAGE,
                found=False,
                hits=hits,
                provider=self.provider.name,
                session_id=session_id,
                confidence_reason="low retrieval confidence",
            )
        else:
            pack = build_context_pack(hits, self.token_budget)
            raw = self.provider.complete(build_request(question, pack, history))
            if is_not_found(raw):
                answer = Answer(
                    question=question,
                    text=NOT_FOUND_MESSAGE,
                    found=False,
                    hits=hits,
                    context=pack.items,
                    provider=self.provider.name,
                    session_id=session_id,
                    confidence_reason="model found no answer in the context",
                )
            else:
                text, citations = check_answer(raw, self.store.get_files(repo.id), pack)
                answer = Answer(
                    question=question,
                    text=text,
                    found=True,
                    citations=citations,
                    hits=hits,
                    context=pack.items,
                    provider=self.provider.name,
                    session_id=session_id,
                    confidence_reason=reason,
                )
        if session_id:
            self.memory.record(session_id, answer)
        return answer
