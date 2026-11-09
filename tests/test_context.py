from repo_qa.models import ChunkKind, SearchHit, StoredChunk
from repo_qa.retrieval.context import build_context_pack, item_tokens, merge_hits

_FILE = [f"line {i}" for i in range(1, 201)]  # a virtual file: line N reads "line N"


def hit(cid: int, start: int, end: int, score: float, path: str = "a.py", sym=None) -> SearchHit:
    text = "\n".join(_FILE[start - 1 : end])
    chunk = StoredChunk(
        id=cid,
        repo_id=1,
        path=path,
        language="python",
        kind=ChunkKind.FUNCTION,
        symbol=sym,
        start_line=start,
        end_line=end,
        text=text,
        content_hash=f"h{cid}",
        blob_hash="b",
    )
    return SearchHit(chunk=chunk, score=score)


def test_overlapping_hits_merge_into_one_region() -> None:
    items = merge_hits([hit(1, 1, 10, 0.5, sym="f"), hit(2, 5, 20, 0.9, sym="g")])
    assert len(items) == 1
    item = items[0]
    assert (item.start_line, item.end_line) == (1, 20)
    assert item.text == "\n".join(_FILE[0:20])
    assert item.score == 0.9 and item.symbols == ["f", "g"] and item.chunk_ids == [1, 2]


def test_adjacent_hits_merge_but_gaps_do_not() -> None:
    items = merge_hits([hit(1, 1, 10, 0.5), hit(2, 11, 15, 0.4), hit(3, 30, 40, 0.3)])
    assert [(i.start_line, i.end_line) for i in items] == [(1, 15), (30, 40)]


def test_same_lines_in_different_files_stay_separate() -> None:
    items = merge_hits([hit(1, 1, 5, 0.5, "a.py"), hit(2, 1, 5, 0.4, "b.py")])
    assert sorted(i.path for i in items) == ["a.py", "b.py"]


def test_duplicate_hits_are_deduped() -> None:
    items = merge_hits([hit(1, 1, 5, 0.5), hit(1, 1, 5, 0.5)])
    assert len(items) == 1 and items[0].chunk_ids == [1]


def test_budget_keeps_best_items_in_score_order() -> None:
    hits = [hit(1, 10, 19, 0.2), hit(2, 50, 59, 0.9), hit(3, 70, 79, 0.5)]  # equal sizes
    one = item_tokens(merge_hits([hits[1]])[0])
    pack = build_context_pack(hits, token_budget=one * 2 + 1)
    assert [i.start_line for i in pack.items] == [50, 70]
    assert pack.dropped == 1
    assert pack.token_count <= pack.token_budget


def test_smaller_item_can_fill_remaining_budget() -> None:
    big, small = hit(1, 1, 100, 0.9), hit(2, 150, 152, 0.1)
    medium = hit(3, 120, 140, 0.5)
    budget = item_tokens(merge_hits([big])[0]) + item_tokens(merge_hits([small])[0])
    pack = build_context_pack([big, medium, small], token_budget=budget)
    assert [i.start_line for i in pack.items] == [1, 150]
    assert pack.dropped == 1


def test_oversized_top_item_is_truncated_not_dropped() -> None:
    pack = build_context_pack([hit(1, 1, 200, 0.9)], token_budget=120)
    assert len(pack.items) == 1
    item = pack.items[0]
    assert item.start_line == 1 and 1 < item.end_line < 200
    assert item.text == "\n".join(_FILE[: item.end_line])
    assert pack.token_count <= 120


def test_empty_hits() -> None:
    pack = build_context_pack([], token_budget=1000)
    assert pack.items == [] and pack.token_count == 0
