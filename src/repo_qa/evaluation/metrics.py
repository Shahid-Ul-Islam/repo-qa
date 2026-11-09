"""Retrieval and answer-quality metrics."""

from __future__ import annotations

from collections.abc import Sequence

from repo_qa.models import Citation, CitationStatus


def path_matches(path: str, expected: str) -> bool:
    """Exact file match, or prefix match when ``expected`` names a directory (ends in /)."""
    return path.startswith(expected) if expected.endswith("/") else path == expected


def is_relevant(path: str, expected_paths: Sequence[str]) -> bool:
    return any(path_matches(path, e) for e in expected_paths)


def first_relevant_rank(retrieved: Sequence[str], expected_paths: Sequence[str]) -> int | None:
    """1-based rank of the first retrieved path that is expected, or None."""
    for rank, path in enumerate(retrieved, start=1):
        if is_relevant(path, expected_paths):
            return rank
    return None


def hit_at_k(retrieved: Sequence[str], expected_paths: Sequence[str], k: int) -> bool:
    rank = first_relevant_rank(retrieved[:k], expected_paths)
    return rank is not None


def reciprocal_rank(retrieved: Sequence[str], expected_paths: Sequence[str], k: int) -> float:
    rank = first_relevant_rank(retrieved[:k], expected_paths)
    return 0.0 if rank is None else 1.0 / rank


def substring_hit(texts: Sequence[str], substrings: Sequence[str]) -> bool | None:
    """Every expected substring occurs (case-insensitively) in some retrieved text."""
    if not substrings:
        return None
    haystack = "\n".join(texts).lower()
    return all(s.lower() in haystack for s in substrings)


def citation_precision(
    citations: Sequence[Citation], expected_paths: Sequence[str]
) -> float | None:
    """Share of surviving citations (verified or ungrounded) that point at expected files.

    Invalid citations were already dropped by verification and count against nothing here;
    an answer with no citations has no precision (None).
    """
    kept = [c for c in citations if c.status is not CitationStatus.INVALID]
    if not kept:
        return None
    good = sum(
        1
        for c in kept
        if c.status is CitationStatus.VERIFIED and is_relevant(c.path, expected_paths)
    )
    return good / len(kept)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
