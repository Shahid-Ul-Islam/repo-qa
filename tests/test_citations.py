import pytest

from repo_qa.answering.citations import (
    INVALID_MARK,
    check_answer,
    parse_citations,
    verify_citations,
)
from repo_qa.models import CitationStatus, ContextItem, ContextPack, FileRecord


def _locs(text: str) -> list[tuple[str, int, int]]:
    return [(c.path, c.start_line, c.end_line) for c in parse_citations(text)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("See src/a.py:12-40.", [("src/a.py", 12, 40)]),
        ("See `src/a.py:12`", [("src/a.py", 12, 12)]),
        ("[src/a.py:3-5] and (lib/b.ts:7\u20139)", [("src/a.py", 3, 5), ("lib/b.ts", 7, 9)]),
        ("src/a.py:L10-L12", [("src/a.py", 10, 12)]),
        ("src/a.py#L10-L12", [("src/a.py", 10, 12)]),
        ("./src/a.py:1-2", [("src/a.py", 1, 2)]),
        ("src\\win\\a.cs:4-6", [("src/win/a.cs", 4, 6)]),
        ("Dockerfile:3", [("Dockerfile", 3, 3)]),
        ("dupe a.py:1-2 and again a.py:1-2", [("a.py", 1, 2)]),
        ("src/a.py: 12 - 40", []),  # space after the colon is not a citation
    ],
)
def test_parse_citations(text: str, expected) -> None:
    assert _locs(text) == expected


@pytest.mark.parametrize(
    "text",
    ["listen on localhost:8080", "at 10:30 today", "version 1.2:3abc", "ratio 16:9", "http://x:80"],
)
def test_parse_ignores_non_citations(text: str) -> None:
    assert _locs(text) == []


FILES = {
    "src/shop/cart.py": FileRecord(
        path="src/shop/cart.py", language="python", size=1, line_count=40, blob_hash="b"
    ),
    "src/shop/payment.py": FileRecord(
        path="src/shop/payment.py", language="python", size=1, line_count=20, blob_hash="b"
    ),
    "tests/payment.py": FileRecord(
        path="tests/payment.py", language="python", size=1, line_count=20, blob_hash="b"
    ),
}
PACK = ContextPack(
    items=[
        ContextItem(
            path="src/shop/cart.py",
            language="python",
            start_line=10,
            end_line=20,
            text="",
            score=1.0,
        )
    ],
    token_count=10,
    token_budget=100,
)


def _verify(text: str):
    return verify_citations(parse_citations(text), FILES, PACK)


def test_verified_when_overlapping_context() -> None:
    [c] = _verify("src/shop/cart.py:18-25")
    assert c.status is CitationStatus.VERIFIED and c.reason is None


def test_ungrounded_when_outside_context() -> None:
    [c] = _verify("src/shop/cart.py:30-35")
    assert c.status is CitationStatus.UNGROUNDED


def test_invalid_path_and_range() -> None:
    missing, past_end, inverted, zero = _verify(
        "nope/x.py:1-2 src/shop/cart.py:35-41 src/shop/cart.py:9-3 src/shop/cart.py:0-2"
    )
    assert missing.status is CitationStatus.INVALID and "not in index" in missing.reason
    assert past_end.status is CitationStatus.INVALID and "40 lines" in past_end.reason
    assert inverted.status is CitationStatus.INVALID
    assert zero.status is CitationStatus.INVALID


def test_unique_suffix_resolves_ambiguous_does_not() -> None:
    [cart] = _verify("cart.py:12-14")
    assert cart.path == "src/shop/cart.py" and cart.status is CitationStatus.VERIFIED
    [pay] = _verify("payment.py:1-2")  # two files end with payment.py
    assert pay.status is CitationStatus.INVALID


def test_check_answer_drops_and_flags_invalid() -> None:
    text = "Items are summed in src/shop/cart.py:12-14, see also ghost.py:1-3."
    flagged, kept = check_answer(text, FILES, PACK)
    assert [c.location for c in kept] == ["src/shop/cart.py:12-14"]
    assert f"ghost.py:1-3 {INVALID_MARK}" in flagged
    assert "cart.py:12-14 " + INVALID_MARK not in flagged
