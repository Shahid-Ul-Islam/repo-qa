import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from repo_qa.cli import app

runner = CliRunner()


@pytest.fixture
def env(tmp_path: Path, monkeypatch) -> Path:
    db = tmp_path / "cli.db"
    monkeypatch.setenv("REPO_QA_DB", str(db))
    monkeypatch.setenv("REPO_QA_EMBEDDER", "mock")
    monkeypatch.setenv("REPO_QA_LLM", "mock")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.chdir(tmp_path)  # not inside any indexed repo
    return db


def _invoke(*args: str):
    result = runner.invoke(app, list(args))
    return result


def test_index_then_status(env, sample_repo) -> None:
    result = _invoke("index", str(sample_repo))
    assert result.exit_code == 0, result.output
    assert "Indexed shop" in result.output

    again = _invoke("index", str(sample_repo))
    assert "0 / 0 / 4 / 0" in again.output  # everything unchanged

    status = _invoke("status", "--json")
    data = json.loads(status.output)
    assert data[0]["repo"]["name"] == "shop" and data[0]["file_count"] == 4


def test_search_json_and_filters(env, sample_repo) -> None:
    _invoke("index", str(sample_repo))
    out = _invoke("search", "credit card", "--json", "-k", "3")
    assert out.exit_code == 0, out.output
    hits = json.loads(out.output)
    assert hits[0]["chunk"]["symbol"] == "charge_credit_card"
    out = _invoke("search", "credit card", "--json", "--include", "docs/*")
    assert {h["chunk"]["path"] for h in json.loads(out.output)} == {"docs/guide.md"}
    table = _invoke("search", "credit card")
    assert "src/shop/payment.py:1-5" in table.output


def test_ask_found_and_not_found(env, sample_repo) -> None:
    _invoke("index", str(sample_repo))
    out = _invoke("ask", "How do we charge a credit card?")
    assert out.exit_code == 0, out.output
    assert "src/shop/payment.py:1-5" in out.output and "✓" in out.output

    data = json.loads(_invoke("ask", "charge credit card", "--json").output)
    assert data["found"] and data["citations"][0]["status"] == "verified"

    missing = _invoke("ask", "What is the capital of France?")
    assert missing.exit_code == 2 and "Not found in the repo" in missing.output


def test_repo_resolution_errors(env, sample_repo, tmp_path) -> None:
    out = _invoke("ask", "anything")
    assert out.exit_code == 1 and "No repositories indexed" in out.output
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.py").write_text("def x():\n    pass\n")
    _invoke("index", str(sample_repo))
    _invoke("index", str(other))
    ambiguous = _invoke("search", "x")
    assert ambiguous.exit_code == 1 and "--repo" in ambiguous.output
    assert _invoke("search", "x", "--repo", "other", "--json").exit_code == 0
    assert _invoke("search", "x", "--repo", str(other), "--json").exit_code == 0
    assert _invoke("search", "x", "--repo", "nope").exit_code == 1


def test_embedder_mismatch_is_reported(env, sample_repo, monkeypatch) -> None:
    _invoke("index", str(sample_repo))
    out = _invoke("search", "cart", "--embedder", "fastembed")
    assert out.exit_code == 1 and "Re-run `repo-qa index`" in out.output


def test_index_rejects_missing_dir(env, tmp_path) -> None:
    out = _invoke("index", str(tmp_path / "missing"))
    assert out.exit_code == 1 and "not a directory" in out.output
