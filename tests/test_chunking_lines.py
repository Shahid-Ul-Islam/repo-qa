import pytest

from repo_qa.chunking.base import chunk_file
from repo_qa.chunking.lines import markdown_chunks, split_lines, window_chunks
from repo_qa.models import ChunkKind


def test_split_lines_counts_like_editors() -> None:
    assert split_lines("a\nb\n") == ["a", "b"]
    assert split_lines("a\nb") == ["a", "b"]
    assert split_lines("a\x0cb\n") == ["a\x0cb"]  # form feed is not a line break
    assert split_lines("") == []


def test_windows_cover_every_line_with_overlap() -> None:
    lines = [f"line {i}" for i in range(1, 131)]
    chunks = window_chunks(lines, language="text", window=60, overlap=10)
    assert [(c.start_line, c.end_line) for c in chunks] == [(1, 60), (51, 110), (101, 130)]
    assert chunks[0].text.splitlines()[0] == "line 1"
    assert chunks[-1].text.splitlines()[-1] == "line 130"
    assert all(c.kind is ChunkKind.WINDOW for c in chunks)


def test_windows_trim_blank_edges_and_skip_blank_windows() -> None:
    lines = ["", "", "x", "", "", "", ""]
    chunks = window_chunks(lines, language="text", window=3, overlap=1)
    assert [(c.start_line, c.end_line) for c in chunks] == [(3, 3), (3, 3)]


def test_window_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        window_chunks(["a"], language="text", window=5, overlap=5)


MD = """\
Intro paragraph.

# Install
Run pip.

## From source
```bash
# not a heading
git clone x
```

# Usage
Ask things.
"""


def test_markdown_sections_and_breadcrumbs() -> None:
    chunks = markdown_chunks(split_lines(MD))
    got = [(c.symbol, c.start_line, c.end_line) for c in chunks]
    assert got == [
        (None, 1, 1),
        ("Install", 3, 4),
        ("Install > From source", 6, 10),
        ("Usage", 12, 13),
    ]
    assert "# not a heading" in chunks[2].text
    assert all(c.kind is ChunkKind.SECTION for c in chunks)


def test_markdown_long_section_is_windowed_with_symbol() -> None:
    lines = ["# Big"] + [f"text {i}" for i in range(200)]
    chunks = markdown_chunks(lines, max_lines=50)
    assert len(chunks) > 1
    assert all(c.symbol == "Big" for c in chunks)
    assert chunks[-1].end_line == 201


def test_chunk_file_normalizes_crlf_and_dispatches() -> None:
    chunks = chunk_file("# T\r\nbody\r\n", "markdown")
    assert [(c.symbol, c.text) for c in chunks] == [("T", "# T\nbody")]
    assert chunk_file("", "text") == []
    assert chunk_file("a\nb\n", "yaml")[0].kind is ChunkKind.WINDOW
