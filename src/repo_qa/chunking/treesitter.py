"""Syntactic chunking with tree-sitter.

Every function/method becomes its own chunk (with leading comments, decorators and
attributes). Lines not covered by a function — imports, constants, class headers and
fields — are grouped into gap chunks tagged with the innermost enclosing type (``class``
kind) or no symbol (``module`` kind), so every non-blank line of a file is indexed exactly
once. Units longer than ``max_lines`` are split into windows that keep their symbol.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

from tree_sitter import Language, Node, Parser

from repo_qa.chunking.lines import make_chunk, window_chunks
from repo_qa.models import ChunkKind, CodeChunk


@dataclass(frozen=True)
class LangSpec:
    load: Callable[[], object]
    functions: frozenset[str]
    containers: frozenset[str]
    comments: frozenset[str] = frozenset({"comment"})
    var_decls: frozenset[str] = frozenset()  # `const f = () => {}` style definitions
    wrappers: frozenset[str] = frozenset()  # nodes whose child is the real definition


def _ts(name: str) -> Callable[[], object]:
    def load() -> object:
        import tree_sitter_typescript

        return getattr(tree_sitter_typescript, name)()

    return load


def _mod(module: str) -> Callable[[], object]:
    def load() -> object:
        import importlib

        return importlib.import_module(module).language()

    return load


_JS_FUNCS = frozenset(
    {"function_declaration", "generator_function_declaration", "method_definition"}
)
_JS_VARS = frozenset({"lexical_declaration", "variable_declaration"})
_TS_CONTAINERS = frozenset(
    {
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "enum_declaration",
        "type_alias_declaration",
    }
)

SPECS: dict[str, LangSpec] = {
    "python": LangSpec(
        load=_mod("tree_sitter_python"),
        functions=frozenset({"function_definition"}),
        containers=frozenset({"class_definition"}),
        wrappers=frozenset({"decorated_definition"}),
    ),
    "javascript": LangSpec(
        load=_mod("tree_sitter_javascript"),
        functions=_JS_FUNCS,
        containers=frozenset({"class_declaration"}),
        var_decls=_JS_VARS,
        wrappers=frozenset({"export_statement"}),
    ),
    "typescript": LangSpec(
        load=_ts("language_typescript"),
        functions=_JS_FUNCS,
        containers=_TS_CONTAINERS,
        var_decls=_JS_VARS,
        wrappers=frozenset({"export_statement"}),
    ),
    "tsx": LangSpec(
        load=_ts("language_tsx"),
        functions=_JS_FUNCS,
        containers=_TS_CONTAINERS,
        var_decls=_JS_VARS,
        wrappers=frozenset({"export_statement"}),
    ),
    "csharp": LangSpec(
        load=_mod("tree_sitter_c_sharp"),
        functions=frozenset(
            {
                "method_declaration",
                "constructor_declaration",
                "destructor_declaration",
                "operator_declaration",
                "local_function_statement",
            }
        ),
        containers=frozenset(
            {
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
                "record_declaration",
                "enum_declaration",
            }
        ),
    ),
    "java": LangSpec(
        load=_mod("tree_sitter_java"),
        functions=frozenset(
            {"method_declaration", "constructor_declaration", "compact_constructor_declaration"}
        ),
        containers=frozenset(
            {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
                "annotation_type_declaration",
            }
        ),
        comments=frozenset({"line_comment", "block_comment"}),
    ),
    "go": LangSpec(
        load=_mod("tree_sitter_go"),
        functions=frozenset({"function_declaration", "method_declaration"}),
        containers=frozenset({"type_declaration"}),
    ),
    "rust": LangSpec(
        load=_mod("tree_sitter_rust"),
        functions=frozenset({"function_item"}),
        containers=frozenset({"struct_item", "enum_item", "union_item", "trait_item", "impl_item"}),
        comments=frozenset({"line_comment", "block_comment", "attribute_item"}),
    ),
}


@cache
def _parser(language: str) -> Parser:
    return Parser(Language(SPECS[language].load()))


def supports(language: str) -> bool:
    return language in SPECS


@dataclass
class _Unit:
    start: int  # 1-based inclusive
    end: int
    kind: ChunkKind
    symbol: str


@dataclass
class _Collector:
    spec: LangSpec
    units: list[_Unit] = field(default_factory=list)
    containers: list[tuple[int, int, str]] = field(default_factory=list)  # DFS order


def _start_line(node: Node) -> int:
    return node.start_point.row + 1


def _end_line(node: Node) -> int:
    row, col = node.end_point.row, node.end_point.column
    # A node ending at column 0 actually ends on the previous line.
    return max(row if col == 0 else row + 1, _start_line(node))


def _text(node: Node | None) -> str:
    return node.text.decode("utf-8", "replace") if node is not None and node.text else ""


_GENERIC_ARGS = re.compile(r"<.*>|\[.*\]")


def _clean_type(name: str) -> str:
    return _GENERIC_ARGS.sub("", name).lstrip("*&").strip()


def _name(node: Node, language: str) -> str:
    if language == "go" and node.type == "type_declaration":
        for child in node.named_children:
            if child.type in ("type_spec", "type_alias"):
                return _text(child.child_by_field_name("name"))
    if language == "rust" and node.type == "impl_item":
        return _clean_type(_text(node.child_by_field_name("type")))
    name = node.child_by_field_name("name")
    if name is not None:
        return _text(name)
    return node.type


def _go_receiver(node: Node) -> str | None:
    receiver = node.child_by_field_name("receiver")
    if receiver is None:
        return None
    for param in receiver.named_children:
        typ = param.child_by_field_name("type")
        if typ is not None:
            return _clean_type(_text(typ))
    return None


def _var_function_name(node: Node) -> str | None:
    """Name for ``const f = () => {}`` / ``var f = function () {}``; else None."""
    declarators = [c for c in node.named_children if c.type == "variable_declarator"]
    if len(declarators) != 1:
        return None
    value = declarators[0].child_by_field_name("value")
    if value is None or value.type not in ("arrow_function", "function_expression", "function"):
        return None
    return _text(declarators[0].child_by_field_name("name")) or None


def _outer(node: Node, spec: LangSpec) -> Node:
    """The node whose range a definition really occupies (export/decorator wrappers)."""
    while node.parent is not None and node.parent.type in spec.wrappers:
        node = node.parent
    return node


def _leading_start(node: Node, spec: LangSpec) -> int:
    """Extend upwards over directly-adjacent doc comments / attributes."""
    start = _start_line(node)
    prev = node.prev_named_sibling
    while prev is not None and prev.type in spec.comments and _end_line(prev) >= start - 1:
        start = _start_line(prev)
        prev = prev.prev_named_sibling
    return start


def _visit(node: Node, scope: list[str], col: _Collector, language: str) -> None:
    spec = col.spec
    for child in node.named_children:
        target = child
        if child.type in spec.wrappers and child.type == "decorated_definition":
            target = child.child_by_field_name("definition") or child

        if target.type in spec.functions:
            outer = _outer(target, spec)
            name = _name(target, language)
            qualifier = scope[-1] if scope else None
            if language == "go" and target.type == "method_declaration":
                qualifier = _go_receiver(target)
            kind = ChunkKind.METHOD if qualifier else ChunkKind.FUNCTION
            symbol = f"{qualifier}.{name}" if qualifier else name
            col.units.append(_Unit(_leading_start(outer, spec), _end_line(outer), kind, symbol))
        elif target.type in spec.var_decls and (fname := _var_function_name(target)):
            outer = _outer(target, spec)
            symbol = f"{scope[-1]}.{fname}" if scope else fname
            kind = ChunkKind.METHOD if scope else ChunkKind.FUNCTION
            col.units.append(_Unit(_leading_start(outer, spec), _end_line(outer), kind, symbol))
        elif target.type in spec.containers:
            outer = _outer(target, spec)
            name = _name(target, language)
            qualified = f"{scope[-1]}.{name}" if scope else name
            col.containers.append((_leading_start(outer, spec), _end_line(outer), qualified))
            _visit(target, [*scope, qualified], col, language)
        else:
            _visit(child, scope, col, language)


_PUNCT_ONLY = re.compile(r"^[\s{}()\[\];,]*$")


def chunk_syntactic(
    text: str, lines: list[str], language: str, *, max_lines: int
) -> list[CodeChunk]:
    spec = SPECS[language]
    tree = _parser(language).parse(text.encode("utf-8"))
    col = _Collector(spec)
    _visit(tree.root_node, [], col, language)

    n = len(lines)
    covered = [False] * (n + 2)
    owner: list[str | None] = [None] * (n + 2)
    for start, end, name in col.containers:  # DFS order: inner containers paint last
        for i in range(start, min(end, n) + 1):
            owner[i] = name

    chunks: list[CodeChunk] = []
    for u in col.units:
        end = min(u.end, n)
        for i in range(u.start, end + 1):
            covered[i] = True
        chunks.extend(_emit(lines, u.start, end, language, u.kind, u.symbol, max_lines))

    # Gap chunks: runs of uncovered lines sharing the same enclosing type.
    i = 1
    while i <= n:
        if covered[i]:
            i += 1
            continue
        j = i
        while j + 1 <= n and not covered[j + 1] and owner[j + 1] == owner[i]:
            j += 1
        body = "\n".join(lines[i - 1 : j])
        if not _PUNCT_ONLY.match(body):
            kind = ChunkKind.CLASS if owner[i] else ChunkKind.MODULE
            chunks.extend(_emit(lines, i, j, language, kind, owner[i], max_lines))
        i = j + 1

    chunks.sort(key=lambda c: (c.start_line, c.end_line))
    return chunks


def _emit(
    lines: list[str],
    start: int,
    end: int,
    language: str,
    kind: ChunkKind,
    symbol: str | None,
    max_lines: int,
) -> list[CodeChunk]:
    if end - start + 1 > max_lines:
        return window_chunks(
            lines, language=language, start=start, end=end, kind=kind, symbol=symbol
        )
    chunk = make_chunk(lines, start, end, language=language, kind=kind, symbol=symbol)
    return [chunk] if chunk is not None else []
