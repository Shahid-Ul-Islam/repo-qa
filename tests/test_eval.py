import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from repo_qa.answering.providers.mock import MockProvider
from repo_qa.answering.qa import QAEngine
from repo_qa.cli import app
from repo_qa.embeddings.mock import MockEmbedder
from repo_qa.evaluation.metrics import (
    citation_precision,
    first_relevant_rank,
    hit_at_k,
    path_matches,
    reciprocal_rank,
    substring_hit,
)
from repo_qa.evaluation.runner import EvalFileError, load_cases, run_eval
from repo_qa.indexing.indexer import Indexer
from repo_qa.models import Citation, CitationStatus
from repo_qa.retrieval.search import Searcher
from repo_qa.store.repository import Store

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SELF_EVAL = PROJECT_ROOT / "evals" / "repo_qa_self.jsonl"


# ----------------------------------------------------------------------------- metrics
def test_rank_metrics() -> None:
    retrieved = ["a.py", "b.py", "c.py", "b.py"]
    assert first_relevant_rank(retrieved, ["b.py"]) == 2
    assert first_relevant_rank(retrieved, ["z.py"]) is None
    assert hit_at_k(retrieved, ["c.py"], 3) and not hit_at_k(retrieved, ["c.py"], 2)
    assert reciprocal_rank(retrieved, ["c.py"], 5) == pytest.approx(1 / 3)
    assert reciprocal_rank(retrieved, ["c.py"], 2) == 0.0
    assert reciprocal_rank(retrieved, ["a.py", "c.py"], 5) == 1.0


def test_directory_expectations() -> None:
    assert path_matches("src/x/y.py", "src/x/")
    assert not path_matches("src/xy.py", "src/x")
    assert not path_matches("src/x/y.py", "src/x")


def test_substring_hit() -> None:
    assert substring_hit(["def Foo(): ...", "bar"], ["foo", "BAR"]) is True
    assert substring_hit(["def foo"], ["foo", "baz"]) is False
    assert substring_hit(["x"], []) is None


def _cit(path: str, status: CitationStatus) -> Citation:
    return Citation(raw=path, path=path, start_line=1, end_line=2, status=status)


def test_citation_precision() -> None:
    cits = [
        _cit("a.py", CitationStatus.VERIFIED),
        _cit("b.py", CitationStatus.VERIFIED),
        _cit("a.py", CitationStatus.UNGROUNDED),  # right file, but not grounded: not credited
        _cit("ghost.py", CitationStatus.INVALID),  # already dropped upstream: ignored
    ]
    assert citation_precision(cits, ["a.py"]) == pytest.approx(1 / 3)
    assert citation_precision([], ["a.py"]) is None


# ------------------------------------------------------------------------------ loader
def test_load_cases_skips_comments_and_reports_bad_lines(tmp_path: Path) -> None:
    good = tmp_path / "good.jsonl"
    good.write_text('// comment\n\n{"question": "q", "expected_paths": ["a.py"]}\n')
    assert [c.question for c in load_cases(good)] == ["q"]
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"question": "q", "expected_paths": ["a.py"]}\n{"question": "no paths"}\n')
    with pytest.raises(EvalFileError, match=r"bad.jsonl:2"):
        load_cases(bad)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("# nothing\n")
    with pytest.raises(EvalFileError):
        load_cases(empty)


def test_shipped_eval_set_is_valid() -> None:
    cases = load_cases(SELF_EVAL)
    assert len(cases) >= 20
    for case in cases:
        for path in case.expected_paths:
            assert (PROJECT_ROOT / path).exists(), f"{path} in eval set does not exist"


# ------------------------------------------------------------- self-eval (mock providers)
@pytest.fixture(scope="module")
def self_index(tmp_path_factory):
    store = Store.open(tmp_path_factory.mktemp("selfeval") / "index.db")
    embedder = MockEmbedder()
    repo, _ = Indexer(store, embedder).index(PROJECT_ROOT)
    yield store, embedder, repo
    store.close()


@pytest.mark.parametrize(
    ("exclude", "min_hit_rate", "min_mrr"),
    [
        # Primary gate: questions are about source code. Test files restate the identifiers
        # of the code they test, which the lexical mock embedder cannot tell apart.
        # Baseline when set: hit-rate@5 0.955, MRR@5 0.667.
        (["evals/*", "tests/*"], 0.90, 0.60),
        # Sanity floor over everything indexed (tests and docs compete for ranks).
        # Baseline when set: hit-rate@5 0.773, MRR@5 0.538.
        (["evals/*"], 0.70, 0.45),
    ],
    ids=["source", "everything"],
)
def test_self_eval_retrieval_quality(self_index, exclude, min_hit_rate, min_mrr) -> None:
    """Regression floors for retrieval on repo-qa's own code (offline, mock embedder)."""
    store, embedder, repo = self_index
    report = run_eval(load_cases(SELF_EVAL), repo, Searcher(store, embedder), k=5, exclude=exclude)
    assert report.hit_rate >= min_hit_rate, report.model_dump_json(indent=2)
    assert report.mrr >= min_mrr, report.model_dump_json(indent=2)
    assert report.citation_precision is None  # no answers requested


def test_self_eval_with_answers(self_index) -> None:
    store, embedder, repo = self_index
    searcher = Searcher(store, embedder)
    engine = QAEngine(store, searcher, MockProvider())
    cases = load_cases(SELF_EVAL)[:5]
    report = run_eval(cases, repo, searcher, k=5, engine=engine, exclude=["evals/*"])
    assert report.citation_precision is not None
    assert 0.0 < report.citation_precision <= 1.0


def test_eval_cli(tmp_path: Path, monkeypatch, sample_repo: Path) -> None:
    monkeypatch.setenv("REPO_QA_DB", str(tmp_path / "e.db"))
    monkeypatch.setenv("REPO_QA_EMBEDDER", "mock")
    monkeypatch.setenv("REPO_QA_LLM", "mock")
    monkeypatch.setenv("COLUMNS", "200")
    evalset = tmp_path / "shop.jsonl"
    evalset.write_text(
        '{"question": "charge a credit card", "expected_paths": ["src/shop/payment.py"]}\n'
        '{"question": "cart total price", "expected_paths": ["src/shop/cart.py"],'
        ' "expected_substrings": ["total_price"]}\n'
    )
    runner = CliRunner()
    assert runner.invoke(app, ["index", str(sample_repo)]).exit_code == 0
    out = runner.invoke(app, ["eval", str(evalset), "--json"])
    assert out.exit_code == 0, out.output
    report = json.loads(out.output)
    assert report["hit_rate"] == 1.0 and report["citation_precision"] is None
    out = runner.invoke(app, ["eval", str(evalset), "--answers", "--json"])
    assert json.loads(out.output)["citation_precision"] is not None
    table = runner.invoke(app, ["eval", str(evalset), "-k", "1", "--min-hit-rate", "1.01"])
    assert table.exit_code == 1 and "hit-rate@1" in table.output
