import numpy as np
import pytest

from repo_qa.embeddings.mock import MockEmbedder
from repo_qa.retrieval.fusion import PathFilter, rrf
from repo_qa.retrieval.keyword import build_match_query, keyword_search
from repo_qa.retrieval.search import IndexMismatchError, Searcher
from repo_qa.retrieval.vector import VectorIndex


def test_rrf_scores_and_order() -> None:
    scores = rrf([["a", "b", "c"], ["c", "a"]], k=60)
    assert scores["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert scores["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["b"] == pytest.approx(1 / 62)
    assert sorted(scores, key=scores.get, reverse=True) == ["a", "c", "b"]


def test_rrf_rewards_agreement_over_single_top_rank() -> None:
    # 'x' is 2nd in both lists; 'y' is 1st in one list only.
    scores = rrf([["y", "x"], ["z", "x"]])
    assert scores["x"] > scores["y"]


def test_rrf_empty() -> None:
    assert rrf([[], []]) == {}


@pytest.mark.parametrize(
    ("include", "exclude", "path", "expected"),
    [
        ([], [], "a/b.py", True),
        (["src/*"], [], "src/deep/x.py", True),
        (["src/*"], [], "web/x.ts", False),
        (["*.py"], [], "src/a.py", True),
        (["**/*.md"], [], "README.md", True),
        ([], ["docs/*"], "docs/guide.md", False),
        (["src/*"], ["*test*"], "src/test_a.py", False),
    ],
)
def test_path_filter(include, exclude, path, expected) -> None:
    assert PathFilter(include, exclude)(path) is expected


@pytest.mark.parametrize(
    "query",
    ['"unbalanced', "NEAR(a b)", "foo -bar", "col:value", "a* OR", "AND", "(((", "x' OR 1=1 --"],
)
def test_match_query_is_always_valid_fts(store, query) -> None:
    repo = store.upsert_repo("/r", "r")
    match = build_match_query(query)
    if match is not None:
        store.fts_search(repo.id, match, 5)  # must not raise


def test_match_query_shape() -> None:
    assert (
        build_match_query("How does parseRoute work?")
        == '"parseroute" OR "parse" OR "route" OR "work"'
    )
    assert build_match_query("what is the") is None


def test_vector_index_topk_and_mask() -> None:
    ids = np.array([10, 20, 30], dtype=np.int64)
    m = np.eye(3, dtype=np.float32)
    idx = VectorIndex(ids, m)
    q = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    assert [cid for cid, _ in idx.search(q, 2)] == [20, 30]
    assert [cid for cid, _ in idx.search(q, 5, allowed={10, 30})] == [30, 10]
    assert VectorIndex(np.zeros(0, dtype=np.int64), np.zeros((0, 0))).search(q, 3) == []
    with pytest.raises(ValueError):
        idx.search(np.ones(4, dtype=np.float32), 1)


def test_keyword_search_finds_split_identifiers(indexed) -> None:
    store, _, repo = indexed
    hits = keyword_search(store, repo.id, "parse route", 5)
    top = store.get_chunk(hits[0][0])
    assert top.symbol == "parseRoute"


def test_hybrid_search_ranks_relevant_chunks(indexed) -> None:
    store, embedder, repo = indexed
    s = Searcher(store, embedder)
    hits = s.search(repo, "charge a credit card payment", top_k=3)
    assert hits[0].chunk.symbol == "charge_credit_card"
    assert hits[0].vector_rank is not None and hits[0].keyword_rank is not None
    assert all(h.score > 0 for h in hits)
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)

    hits = s.search(repo, "total price of the cart items", top_k=3)
    assert hits[0].chunk.symbol == "Cart.total_price"


def test_hybrid_search_respects_filters(indexed) -> None:
    store, embedder, repo = indexed
    s = Searcher(store, embedder)
    hits = s.search(repo, "credit card", top_k=5, include=["docs/*"])
    assert hits and {h.chunk.path for h in hits} == {"docs/guide.md"}
    hits = s.search(repo, "credit card", top_k=5, exclude=["src/*", "docs/*"])
    assert all(h.chunk.path == "web/router.ts" for h in hits)
    assert s.search(repo, "credit card", include=["nothing/*"]) == []


def test_search_refuses_mismatched_embedder(indexed) -> None:
    store, _, repo = indexed
    with pytest.raises(IndexMismatchError):
        Searcher(store, MockEmbedder(dim=32)).search(repo, "cart")
