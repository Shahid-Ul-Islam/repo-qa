from pathlib import Path

import pytest

from repo_qa.chunking.base import chunk_file
from repo_qa.chunking.languages import detect_language
from repo_qa.chunking.lines import split_lines
from repo_qa.models import ChunkKind

FIXTURES = Path(__file__).parent / "fixtures" / "chunking"


def _chunks(name: str):
    path = FIXTURES / name
    return chunk_file(path.read_text(encoding="utf-8"), detect_language(name))


def _units(name: str) -> dict[str, tuple[str, int, int]]:
    """symbol -> (kind, start, end) for function/method chunks."""
    return {
        c.symbol: (c.kind.value, c.start_line, c.end_line)
        for c in _chunks(name)
        if c.kind in (ChunkKind.FUNCTION, ChunkKind.METHOD)
    }


EXPECTED_UNITS = {
    "sample.py": {
        "add": ("function", 8, 10),  # includes the leading comment
        "Greeter.__init__": ("method", 18, 19),
        "Greeter.shout": ("method", 21, 23),  # includes the decorator
        "fetch": ("function", 26, 27),
    },
    "sample.js": {
        "readConfig": ("function", 3, 6),
        "double": ("function", 8, 8),
        "Counter.constructor": ("method", 11, 13),
        "Counter.increment": ("method", 15, 17),
    },
    "sample.ts": {
        "loadUser": ("function", 10, 14),
        "UserService.get": ("method", 19, 21),
        "toName": ("function", 24, 24),
    },
    "sample.tsx": {
        "Header": ("function", 5, 7),
        "Footer": ("function", 9, 9),
    },
    "Sample.cs": {
        "OrderRepository.OrderRepository": ("method", 10, 13),
        "OrderRepository.FindOrder": ("method", 15, 19),
        "IClock.Now": ("method", 24, 24),
    },
    "Sample.java": {
        "Inventory.Inventory": ("method", 9, 11),
        "Inventory.count": ("method", 13, 16),
    },
    "sample.go": {
        "NewItem": ("function", 11, 14),
        "Item.Describe": ("method", 16, 19),  # qualified by its receiver type
    },
    "sample.rs": {
        "Point.norm": ("method", 11, 14),
        "Point.fmt": ("method", 18, 20),
        "origin": ("function", 23, 25),
    },
}

EXPECTED_TYPES = {
    "sample.py": {"Greeter"},
    "sample.js": {"Counter"},
    "sample.ts": {"User", "Id", "UserService"},
    "sample.tsx": {"Props"},
    "Sample.cs": {"OrderRepository", "IClock"},
    "Sample.java": {"Inventory", "Inventory.Kind"},
    "sample.go": {"Item"},
    "sample.rs": {"Point"},
}


@pytest.mark.parametrize("name", sorted(EXPECTED_UNITS))
def test_functions_and_methods(name: str) -> None:
    assert _units(name) == EXPECTED_UNITS[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_TYPES))
def test_type_header_chunks(name: str) -> None:
    types = {c.symbol for c in _chunks(name) if c.kind is ChunkKind.CLASS}
    assert types == EXPECTED_TYPES[name]


@pytest.mark.parametrize("name", sorted(EXPECTED_UNITS))
def test_every_meaningful_line_is_covered_once(name: str) -> None:
    lines = split_lines((FIXTURES / name).read_text(encoding="utf-8"))
    chunks = _chunks(name)
    seen: dict[int, int] = {}
    for c in chunks:
        assert c.text == "\n".join(lines[c.start_line - 1 : c.end_line])
        for i in range(c.start_line, c.end_line + 1):
            seen[i] = seen.get(i, 0) + 1
    for i, line in enumerate(lines, start=1):
        if line.strip().strip("{}();,"):
            assert seen.get(i) == 1, f"line {i} covered {seen.get(i, 0)}x: {line!r}"


def test_module_chunk_holds_imports() -> None:
    first = _chunks("sample.py")[0]
    assert first.kind is ChunkKind.MODULE and "import os" in first.text


def test_oversized_function_is_windowed_keeping_symbol() -> None:
    body = "\n".join(f"    x{i} = {i}" for i in range(300))
    chunks = chunk_file(f"def huge():\n{body}\n", "python", max_lines=100)
    assert len(chunks) > 2
    assert {c.symbol for c in chunks} == {"huge"}
    assert {c.kind for c in chunks} == {ChunkKind.FUNCTION}
    assert chunks[0].start_line == 1 and chunks[-1].end_line == 301


def test_syntax_errors_still_chunk() -> None:
    chunks = chunk_file("def ok():\n    return 1\n\ndef broken(:\n", "python")
    assert any(c.symbol == "ok" for c in chunks)


def test_unsupported_language_falls_back_to_windows() -> None:
    chunks = chunk_file("int main() { return 0; }\n", "c")
    assert [c.kind for c in chunks] == [ChunkKind.WINDOW]
