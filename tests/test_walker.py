import subprocess
from pathlib import Path

import pytest

from repo_qa.chunking.languages import detect_language
from repo_qa.indexing.hashing import content_hash, git_blob_hash
from repo_qa.indexing.walker import git_head, walk


def _make_tree(root: Path) -> None:
    files = {
        "src/app.py": "print('hi')\n",
        "src/util.ts": "export const x = 1;\n",
        "docs/guide.md": "# Guide\n",
        "README": "readme\n",
        "notes.log": "ignored by .gitignore\n",
        "src/gen/out.py": "ignored by nested .gitignore\n",
        "src/gen/keep.py": "kept\n",
        "node_modules/lib/index.js": "vendored\n",
        "package-lock.json": "{}\n",
        "web/app.min.js": "minified\n",
        "image.png": "not a source type\n",
        "big.py": "x = 1\n" * 50_000,
    }
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    (root / "src" / "blob.py").write_bytes(b"abc\0def")
    (root / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (root / "src" / "gen" / ".gitignore").write_text("out.py\n", encoding="utf-8")


EXPECTED = ["README", "docs/guide.md", "src/app.py", "src/gen/keep.py", "src/util.ts"]


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def test_walk_git_repo(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    _git(tmp_path, "init", "-q")
    result = walk(tmp_path, max_bytes=100_000)
    assert result.used_git
    assert sorted(f.path for f in result.files) == EXPECTED
    assert result.skipped["too_large"] == 1
    assert result.skipped["binary"] == 1
    assert result.skipped["lockfile"] == 1
    assert result.skipped["minified"] == 1
    assert result.skipped["vendored"] == 1


def test_walk_plain_directory_matches_git(tmp_path: Path) -> None:
    _make_tree(tmp_path)
    result = walk(tmp_path, max_bytes=100_000)
    assert not result.used_git
    assert sorted(f.path for f in result.files) == EXPECTED
    by_path = {f.path: f for f in result.files}
    assert by_path["src/util.ts"].language == "typescript"
    assert by_path["README"].language == "text"


def test_walk_skips_deleted_tracked_file(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", "a.py")
    (tmp_path / "a.py").unlink()
    result = walk(tmp_path)
    assert result.files == [] and result.skipped["missing"] == 1


def test_git_blob_hash_matches_git(tmp_path: Path) -> None:
    data = b"hello\nworld\n"
    f = tmp_path / "f.txt"
    f.write_bytes(data)
    expected = subprocess.run(
        ["git", "hash-object", "--no-filters", str(f)], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert git_blob_hash(data) == expected


def test_git_head(tmp_path: Path) -> None:
    assert git_head(tmp_path) is None
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x\n")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    head = git_head(tmp_path)
    assert head is not None and len(head) == 40


def test_content_hash_is_stable() -> None:
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abd")


@pytest.mark.parametrize(
    ("path", "lang"),
    [
        ("a/b.py", "python"),
        ("x.TSX", "tsx"),
        ("Main.cs", "csharp"),
        ("Dockerfile", "dockerfile"),
        ("lib.rs", "rust"),
        ("photo.jpg", None),
    ],
)
def test_detect_language(path: str, lang: str | None) -> None:
    assert detect_language(path) == lang
