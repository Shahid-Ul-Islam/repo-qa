"""Text helpers shared by keyword search and the mock embedder."""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD = re.compile(r"[A-Za-z0-9]+")
# camelCase / PascalCase / ACRONYMWord boundaries
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")


def split_identifier(ident: str) -> list[str]:
    """``parseHTTPRequest_v2`` -> ``["parse", "http", "request", "v", "2"]``."""
    parts: list[str] = []
    for piece in ident.split("_"):
        parts.extend(m.group(0).lower() for m in _CAMEL.finditer(piece))
    return parts


def identifier_terms(text: str) -> str:
    """Sub-words of compound identifiers in ``text``, space-separated and de-duplicated.

    Stored in the FTS ``terms`` column so ``http request`` matches ``parseHttpRequest``.
    Plain words are already indexed via the ``text`` column, so only split parts are kept.
    """
    seen: dict[str, None] = {}
    for m in _IDENT.finditer(text):
        ident = m.group(0)
        parts = split_identifier(ident)
        if len(parts) > 1:
            for p in parts:
                seen.setdefault(p, None)
    return " ".join(seen)


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens with compound identifiers expanded into their parts too."""
    out: list[str] = []
    for m in _IDENT.finditer(text):
        ident = m.group(0)
        parts = split_identifier(ident)
        out.append(ident.lower())
        if len(parts) > 1:
            out.extend(parts)
    return out


def words(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def estimate_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 characters per token)."""
    return max(1, (len(text) + 3) // 4)
