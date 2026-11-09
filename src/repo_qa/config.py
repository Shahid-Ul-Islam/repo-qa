"""Runtime settings, read from ``REPO_QA_*`` environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

EmbedderName = Literal["fastembed", "mock"]
LLMName = Literal["mock", "anthropic", "openai"]


def default_db_path() -> Path:
    return Path.home() / ".repo-qa" / "index.db"


class Settings(BaseModel):
    db_path: Path = Field(default_factory=default_db_path)
    embedder: EmbedderName = "fastembed"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    llm: LLMName = "mock"
    llm_model: str | None = None
    top_k: int = Field(default=8, ge=1)
    token_budget: int = Field(default=6000, ge=100)
    max_file_bytes: int = Field(default=1_000_000, ge=1)
    # Minimum top cosine similarity to trust retrieval; None = provider default.
    min_vector_score: float | None = None
    history_turns: int = Field(default=4, ge=0)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, **overrides: object) -> Settings:
        """Build settings from environment variables; explicit overrides win (None is ignored)."""
        env = os.environ if env is None else env
        values: dict[str, object] = {}
        for field in cls.model_fields:
            raw = env.get(f"REPO_QA_{field.upper()}")
            if field == "db_path" and raw in (None, ""):
                raw = env.get("REPO_QA_DB")  # documented short alias
            if raw not in (None, ""):
                values[field] = raw
        if "db_path" in values:
            values["db_path"] = Path(str(values["db_path"])).expanduser()
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls.model_validate(values)
