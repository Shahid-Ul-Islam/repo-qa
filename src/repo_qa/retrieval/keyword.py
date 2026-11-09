"""Keyword search over the FTS5 index."""

from __future__ import annotations

import re

from repo_qa.store.repository import Store
from repo_qa.text import split_identifier

_TOKEN = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "of",
        "on",
        "or",
        "show",
        "the",
        "this",
        "that",
        "to",
        "was",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    ]
)


def build_match_query(query: str) -> str | None:
    """Turn free text into a safe FTS5 expression: quoted terms OR-ed together.

    Quoting every term neutralises FTS5 syntax (``NEAR``, ``-``, ``*``, ``:``, quotes) in user
    input. Compound identifiers also contribute their parts so ``parseRoute`` finds
    ``parse route``.
    """
    terms: dict[str, None] = {}
    for raw in _TOKEN.findall(query):
        lowered = raw.lower()
        if lowered in STOPWORDS:
            continue
        terms.setdefault(lowered, None)
        parts = split_identifier(raw)
        if len(parts) > 1:
            for part in parts:
                if part not in STOPWORDS and len(part) > 1:
                    terms.setdefault(part, None)
    if not terms:
        return None
    return " OR ".join(f'"{t}"' for t in terms)


def keyword_search(store: Store, repo_id: int, query: str, limit: int) -> list[tuple[int, float]]:
    match = build_match_query(query)
    if match is None:
        return []
    return store.fts_search(repo_id, match, limit)
