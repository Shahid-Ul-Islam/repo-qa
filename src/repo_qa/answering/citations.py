"""Parse ``path:start-end`` citations out of model output and verify them against the index.

Verification outcome per citation:

* ``verified``   - the file exists, the range lies inside it, and it overlaps the context
  the model was given.
* ``ungrounded`` - the file and range exist, but the model was not shown those lines, so the
  claim cannot have come from the provided context. Kept, but flagged.
* ``invalid``    - unknown path or a line range outside the file. Dropped from the citation
  list and flagged inline in the answer text.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from repo_qa.models import Citation, CitationStatus, ContextPack, FileRecord

# Path chars exclude whitespace, quotes, brackets and colons. Lines may be "12", "12-40" or
# "L12-L40", with an en/em dash allowed as separator; "#L12-L40" (GitHub style) works too.
CITATION_RE = re.compile(
    r"(?P<path>(?:\.{0,2}/)?[A-Za-z0-9_@+\-][A-Za-z0-9_@+\-./\\]*)"
    r"(?::|#L)L?(?P<start>\d{1,7})(?:\s*[-\u2013\u2014]\s*L?(?P<end>\d{1,7}))?"
    r"(?![\d.]*[A-Za-z0-9])"  # not part of a larger token such as a version or time
)
_EXT = re.compile(r"\.[A-Za-z0-9]{1,10}$")
_SPECIAL_NAMES = frozenset({"dockerfile", "makefile", "readme", "license", "changelog"})


def _normalize_path(path: str) -> str:
    path = path.replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    return path.lstrip("/")


def _looks_like_path(path: str) -> bool:
    name = path.rsplit("/", 1)[-1].lower()
    return "/" in path or bool(_EXT.search(path)) or name in _SPECIAL_NAMES


def parse_citations(text: str) -> list[Citation]:
    """All citation-shaped references in ``text``, de-duplicated, in order of appearance.

    Tokens that do not look like file paths (``localhost:8080``) are ignored.
    """
    seen: set[tuple[str, int, int]] = set()
    out: list[Citation] = []
    for m in CITATION_RE.finditer(text):
        path = _normalize_path(m.group("path"))
        if not path or not _looks_like_path(path):
            continue
        start = int(m.group("start"))
        end = int(m.group("end")) if m.group("end") else start
        key = (path, start, end)
        if key in seen:
            continue
        seen.add(key)
        out.append(Citation(raw=m.group(0), path=path, start_line=start, end_line=end))
    return out


def _resolve(path: str, files: Mapping[str, FileRecord]) -> str | None:
    """Exact path, or a unique suffix match (model wrote ``cart.py`` for ``src/cart.py``)."""
    if path in files:
        return path
    matches = [p for p in files if p.endswith("/" + path)]
    return matches[0] if len(matches) == 1 else None


def verify_citations(
    citations: list[Citation], files: Mapping[str, FileRecord], pack: ContextPack
) -> list[Citation]:
    verified: list[Citation] = []
    for c in citations:
        resolved = _resolve(c.path, files)
        if resolved is None:
            verified.append(
                c.model_copy(
                    update={"status": CitationStatus.INVALID, "reason": "path not in index"}
                )
            )
            continue
        record = files[resolved]
        if not (1 <= c.start_line <= c.end_line <= record.line_count):
            verified.append(
                c.model_copy(
                    update={
                        "path": resolved,
                        "status": CitationStatus.INVALID,
                        "reason": f"lines {c.start_line}-{c.end_line} outside file "
                        f"({record.line_count} lines)",
                    }
                )
            )
            continue
        grounded = any(
            item.path == resolved
            and item.start_line <= c.end_line
            and c.start_line <= item.end_line
            for item in pack.items
        )
        verified.append(
            c.model_copy(
                update={
                    "path": resolved,
                    "status": CitationStatus.VERIFIED if grounded else CitationStatus.UNGROUNDED,
                    "reason": None if grounded else "not in the retrieved context",
                }
            )
        )
    return verified


INVALID_MARK = "[⚠ invalid citation]"


def flag_invalid(text: str, citations: list[Citation]) -> str:
    """Annotate invalid citations inline so readers do not trust them."""
    for c in citations:
        if c.status is CitationStatus.INVALID:
            text = text.replace(c.raw, f"{c.raw} {INVALID_MARK}")
    return text


def check_answer(
    text: str, files: Mapping[str, FileRecord], pack: ContextPack
) -> tuple[str, list[Citation]]:
    """Parse + verify; returns the flagged text and the citations that survived
    (verified and ungrounded 2014 invalid ones are dropped)."""
    citations = verify_citations(parse_citations(text), files, pack)
    kept = [c for c in citations if c.status is not CitationStatus.INVALID]
    return flag_invalid(text, citations), kept
