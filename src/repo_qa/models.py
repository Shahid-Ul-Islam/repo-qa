"""Pydantic domain models shared across indexing, retrieval, answering and evaluation."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class ChunkKind(StrEnum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    MODULE = "module"
    SECTION = "section"
    WINDOW = "window"


class CodeChunk(BaseModel):
    """A chunk as produced by a chunker, before it is tied to a file in the index.

    Line numbers are 1-based and inclusive.
    """

    language: str
    kind: ChunkKind
    symbol: str | None = None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str

    @model_validator(mode="after")
    def _check_range(self) -> CodeChunk:
        if self.end_line < self.start_line:
            raise ValueError(f"end_line {self.end_line} < start_line {self.start_line}")
        return self


class StoredChunk(CodeChunk):
    """A chunk persisted in the index."""

    id: int
    repo_id: int
    path: str
    content_hash: str
    blob_hash: str

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class FileRecord(BaseModel):
    path: str  # POSIX-style, relative to the repo root
    language: str
    size: int
    line_count: int
    blob_hash: str


class RepoRecord(BaseModel):
    id: int
    root_path: str
    name: str
    head_commit: str | None = None
    embed_model: str | None = None
    embed_dim: int | None = None
    indexed_at: datetime | None = None


class RepoStatus(BaseModel):
    repo: RepoRecord
    file_count: int
    chunk_count: int
    embedding_count: int
    languages: dict[str, int]


class IndexStats(BaseModel):
    files_scanned: int = 0
    files_added: int = 0
    files_updated: int = 0
    files_unchanged: int = 0
    files_removed: int = 0
    chunks_written: int = 0
    embeddings_computed: int = 0
    embeddings_reused: int = 0
    duration_s: float = 0.0


class SearchHit(BaseModel):
    chunk: StoredChunk
    score: float
    vector_score: float | None = None
    vector_rank: int | None = None
    keyword_score: float | None = None
    keyword_rank: int | None = None


class ContextItem(BaseModel):
    """A contiguous region of one file selected for the prompt (may merge several chunks)."""

    path: str
    language: str
    start_line: int
    end_line: int
    text: str
    score: float
    symbols: list[str] = Field(default_factory=list)
    chunk_ids: list[int] = Field(default_factory=list)

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class ContextPack(BaseModel):
    items: list[ContextItem]
    token_count: int
    token_budget: int
    dropped: int = 0  # candidate regions that did not fit the budget


class CitationStatus(StrEnum):
    VERIFIED = "verified"  # exists in the repo and overlaps the provided context
    UNGROUNDED = "ungrounded"  # exists in the repo but was not in the provided context
    INVALID = "invalid"  # unknown path or line range outside the file


class Citation(BaseModel):
    raw: str
    path: str
    start_line: int
    end_line: int
    status: CitationStatus = CitationStatus.INVALID
    reason: str | None = None

    @property
    def location(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


class Answer(BaseModel):
    question: str
    text: str
    found: bool
    citations: list[Citation] = Field(default_factory=list)
    hits: list[SearchHit] = Field(default_factory=list)
    context: list[ContextItem] = Field(default_factory=list)
    provider: str
    session_id: str | None = None
    confidence_reason: str | None = None

    @property
    def verified_citations(self) -> list[Citation]:
        return [c for c in self.citations if c.status is CitationStatus.VERIFIED]


class Message(BaseModel):
    role: str  # "user" | "assistant"
    content: str
    citations: list[Citation] = Field(default_factory=list)
    created_at: datetime | None = None


class EvalCase(BaseModel):
    question: str
    expected_paths: list[str] = Field(min_length=1)
    expected_substrings: list[str] = Field(default_factory=list)


class EvalCaseResult(BaseModel):
    question: str
    retrieved_paths: list[str]
    first_relevant_rank: int | None  # 1-based, None if no expected path was retrieved
    hit: bool
    substring_hit: bool | None = None
    citation_precision: float | None = None


class EvalReport(BaseModel):
    k: int
    cases: list[EvalCaseResult]
    hit_rate: float
    mrr: float
    substring_hit_rate: float | None = None
    citation_precision: float | None = None
