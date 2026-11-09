"""Enumerate indexable files in a repository, respecting .gitignore."""

from __future__ import annotations

import os
import subprocess
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import pathspec

from repo_qa.chunking.languages import detect_language

VENDORED_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "bower_components",
        "vendor",
        "third_party",
        "dist",
        "build",
        "out",
        "target",
        "bin",
        "obj",
        ".venv",
        "venv",
        "env",
        "site-packages",
        "__pycache__",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".next",
        ".nuxt",
        "coverage",
        "Pods",
        ".gradle",
        ".idea",
        ".vscode",
        ".repo-qa",
    }
)

LOCKFILES = frozenset(
    {
        "package-lock.json",
        "npm-shrinkwrap.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "bun.lockb",
        "poetry.lock",
        "pipfile.lock",
        "uv.lock",
        "pdm.lock",
        "cargo.lock",
        "go.sum",
        "composer.lock",
        "gemfile.lock",
        "packages.lock.json",
        "podfile.lock",
        "mix.lock",
    }
)

MINIFIED_SUFFIXES = (".min.js", ".min.css", ".map")
BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class SourceFile:
    path: str  # POSIX, relative to the repo root
    abs_path: Path
    language: str
    size: int


@dataclass
class WalkResult:
    files: list[SourceFile] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    used_git: bool = False


def is_git_repo(root: Path) -> bool:
    return _git(root, "rev-parse", "--is-inside-work-tree") == "true"


def git_head(root: Path) -> str | None:
    return _git(root, "rev-parse", "HEAD")


def _git(root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def _git_candidates(root: Path) -> list[str] | None:
    """Tracked + untracked-but-not-ignored files, exactly as git sees them."""
    out = _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
    if out is None:
        return None
    return sorted({p for p in out.split("\0") if p})


def _fs_candidates(root: Path) -> list[str]:
    """Fallback for non-git directories: walk the tree honouring nested .gitignore files."""
    results: list[str] = []
    specs: list[tuple[str, pathspec.GitIgnoreSpec]] = []

    def ignored(rel: str, is_dir: bool) -> bool:
        for prefix, spec in specs:
            if prefix and not rel.startswith(prefix + "/"):
                continue
            sub = rel[len(prefix) + 1 :] if prefix else rel
            if spec.match_file(sub + "/" if is_dir else sub):
                return True
        return False

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        rel_dir = "" if rel_dir == "." else rel_dir
        gi = Path(dirpath) / ".gitignore"
        if gi.is_file():
            lines = gi.read_text(encoding="utf-8", errors="replace").splitlines()
            specs.append((rel_dir, pathspec.GitIgnoreSpec.from_lines(lines)))

        def rel(name: str, base: str = rel_dir) -> str:
            return f"{base}/{name}" if base else name

        dirnames[:] = sorted(
            d for d in dirnames if d not in VENDORED_DIRS and not ignored(rel(d), True)
        )
        results.extend(rel(f) for f in sorted(filenames) if not ignored(rel(f), False))
    return results


def _skip_reason(rel: str, root: Path, max_bytes: int) -> tuple[str | None, SourceFile | None]:
    p = PurePosixPath(rel)
    if any(part in VENDORED_DIRS for part in p.parts[:-1]):
        return "vendored", None
    name = p.name.lower()
    if name in LOCKFILES:
        return "lockfile", None
    if name.endswith(MINIFIED_SUFFIXES):
        return "minified", None
    language = detect_language(rel)
    if language is None:
        return "unsupported_type", None
    abs_path = root / rel
    try:
        size = abs_path.stat().st_size
    except OSError:
        return "missing", None  # tracked but deleted in the working tree
    if not abs_path.is_file():
        return "missing", None
    if size > max_bytes:
        return "too_large", None
    try:
        with abs_path.open("rb") as fh:
            if b"\0" in fh.read(BINARY_SNIFF_BYTES):
                return "binary", None
    except OSError:
        return "unreadable", None
    return None, SourceFile(path=rel, abs_path=abs_path, language=language, size=size)


def walk(
    root: Path, *, max_bytes: int = 1_000_000, candidates: Iterable[str] | None = None
) -> WalkResult:
    """List indexable files under ``root`` with reasons for everything skipped."""
    root = root.resolve()
    result = WalkResult()
    if candidates is None:
        git_files = _git_candidates(root) if is_git_repo(root) else None
        result.used_git = git_files is not None
        candidates = git_files if git_files is not None else _fs_candidates(root)
    for rel in candidates:
        reason, source = _skip_reason(rel, root, max_bytes)
        if reason:
            result.skipped[reason] += 1
        else:
            assert source is not None
            result.files.append(source)
    return result
