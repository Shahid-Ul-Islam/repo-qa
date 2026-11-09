"""Reciprocal rank fusion and path filters."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from fnmatch import fnmatchcase

RRF_K = 60


def rrf(rankings: Iterable[Sequence[Hashable]], k: int = RRF_K) -> dict[Hashable, float]:
    """Reciprocal rank fusion: ``score(d) = Σ 1 / (k + rank_i(d))`` with 1-based ranks.

    Rank-based, so it needs no score calibration between cosine similarity and BM25.
    """
    scores: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


def _glob_match(path: str, pattern: str) -> bool:
    if fnmatchcase(path, pattern):
        return True
    # "**/x" should also match "x" at the repo root
    return pattern.startswith("**/") and fnmatchcase(path, pattern[3:])


class PathFilter:
    """Include/exclude globs over repo-relative POSIX paths (``*`` also crosses ``/``)."""

    def __init__(self, include: Sequence[str] = (), exclude: Sequence[str] = ()) -> None:
        self.include = [p for p in include if p]
        self.exclude = [p for p in exclude if p]

    @property
    def active(self) -> bool:
        return bool(self.include or self.exclude)

    def __call__(self, path: str) -> bool:
        if self.include and not any(_glob_match(path, p) for p in self.include):
            return False
        return not any(_glob_match(path, p) for p in self.exclude)
