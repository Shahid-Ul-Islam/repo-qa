"""Chunker entry point: picks syntactic or line-based chunking per language."""

from __future__ import annotations

from repo_qa.chunking.lines import markdown_chunks, split_lines, window_chunks
from repo_qa.models import CodeChunk

MAX_CHUNK_LINES = 120


def normalize_text(text: str) -> str:
    """Normalise line endings so line numbers are identical on every OS."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def chunk_file(text: str, language: str, *, max_lines: int = MAX_CHUNK_LINES) -> list[CodeChunk]:
    """Split a file's text into chunks; line numbers are 1-based and inclusive."""
    lines = split_lines(normalize_text(text))
    if not lines:
        return []
    if language == "markdown":
        return markdown_chunks(lines, max_lines=max_lines)
    return window_chunks(lines, language=language)
