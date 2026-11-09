from pathlib import Path

import pytest
from pydantic import ValidationError

from repo_qa.config import Settings
from repo_qa.models import ChunkKind, CodeChunk, EvalCase, StoredChunk


def test_chunk_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError):
        CodeChunk(language="python", kind=ChunkKind.FUNCTION, start_line=5, end_line=4, text="x")


def test_stored_chunk_location() -> None:
    chunk = StoredChunk(
        id=1,
        repo_id=1,
        path="src/a.py",
        language="python",
        kind=ChunkKind.FUNCTION,
        symbol="f",
        start_line=3,
        end_line=9,
        text="def f(): ...",
        content_hash="h",
        blob_hash="b",
    )
    assert chunk.location == "src/a.py:3-9"


def test_eval_case_requires_expected_paths() -> None:
    with pytest.raises(ValidationError):
        EvalCase(question="q", expected_paths=[])


def test_settings_from_env_and_overrides(tmp_path: Path) -> None:
    env = {
        "REPO_QA_DB_PATH": str(tmp_path / "x.db"),
        "REPO_QA_TOP_K": "5",
        "REPO_QA_EMBEDDER": "mock",
        "REPO_QA_LLM": "",
    }
    s = Settings.from_env(env, top_k=3, llm_model=None)
    assert s.db_path == tmp_path / "x.db"
    assert s.top_k == 3
    assert s.embedder == "mock"
    assert s.llm == "mock"


def test_settings_rejects_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"REPO_QA_LLM": "nope"})


def test_settings_db_alias(tmp_path: Path) -> None:
    assert Settings.from_env({"REPO_QA_DB": str(tmp_path / "a.db")}).db_path == tmp_path / "a.db"
