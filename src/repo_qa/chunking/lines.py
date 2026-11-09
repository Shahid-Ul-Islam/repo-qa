"""Line-window chunking (fallback for any text) and heading-based Markdown chunking."""

from __future__ import annotations

import re

from repo_qa.models import ChunkKind, CodeChunk

DEFAULT_WINDOW = 60
DEFAULT_OVERLAP = 10

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def split_lines(text: str) -> list[str]:
    """Split on ``\\n`` only, so numbering matches editors and ``git blame``."""
    if not text:
        return []
    lines = text.split("\n")
    if text.endswith("\n"):
        lines.pop()
    return lines


def trim_range(lines: list[str], start: int, end: int) -> tuple[int, int] | None:
    """Shrink a 1-based inclusive range to exclude blank edge lines; None if all blank."""
    while start <= end and not lines[start - 1].strip():
        start += 1
    while end >= start and not lines[end - 1].strip():
        end -= 1
    return (start, end) if start <= end else None


def make_chunk(
    lines: list[str],
    start: int,
    end: int,
    *,
    language: str,
    kind: ChunkKind,
    symbol: str | None = None,
) -> CodeChunk | None:
    trimmed = trim_range(lines, start, end)
    if trimmed is None:
        return None
    s, e = trimmed
    return CodeChunk(
        language=language,
        kind=kind,
        symbol=symbol,
        start_line=s,
        end_line=e,
        text="\n".join(lines[s - 1 : e]),
    )


def window_chunks(
    lines: list[str],
    *,
    language: str,
    start: int = 1,
    end: int | None = None,
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
    kind: ChunkKind = ChunkKind.WINDOW,
    symbol: str | None = None,
) -> list[CodeChunk]:
    """Overlapping fixed-size windows over ``lines[start..end]`` (1-based, inclusive)."""
    if overlap >= window:
        raise ValueError("overlap must be smaller than window")
    end = len(lines) if end is None else end
    chunks: list[CodeChunk] = []
    pos = start
    while pos <= end:
        win_end = min(pos + window - 1, end)
        chunk = make_chunk(lines, pos, win_end, language=language, kind=kind, symbol=symbol)
        if chunk is not None:
            chunks.append(chunk)
        if win_end >= end:
            break
        pos = win_end - overlap + 1
    return chunks


def markdown_chunks(
    lines: list[str], *, language: str = "markdown", max_lines: int = 120
) -> list[CodeChunk]:
    """One chunk per heading section (symbol = heading breadcrumb), windowed if too long.

    Headings inside fenced code blocks are ignored.
    """
    sections: list[tuple[int, str | None]] = []  # (start_line, breadcrumb)
    trail: list[tuple[int, str]] = []
    in_fence = False
    for i, line in enumerate(lines, start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING.match(line)
        if not m:
            continue
        level, title = len(m.group(1)), m.group(2)
        trail = [(lvl, t) for lvl, t in trail if lvl < level] + [(level, title)]
        sections.append((i, " > ".join(t for _, t in trail)))
    if not sections or sections[0][0] != 1:
        sections.insert(0, (1, None))

    chunks: list[CodeChunk] = []
    for idx, (start, crumb) in enumerate(sections):
        end = sections[idx + 1][0] - 1 if idx + 1 < len(sections) else len(lines)
        if end < start:
            continue
        if end - start + 1 > max_lines:
            chunks.extend(
                window_chunks(
                    lines,
                    language=language,
                    start=start,
                    end=end,
                    kind=ChunkKind.SECTION,
                    symbol=crumb,
                )
            )
        else:
            chunk = make_chunk(
                lines, start, end, language=language, kind=ChunkKind.SECTION, symbol=crumb
            )
            if chunk is not None:
                chunks.append(chunk)
    return chunks
