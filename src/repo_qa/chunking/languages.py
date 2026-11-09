"""File-extension → language detection, and which languages get syntactic chunking."""

from __future__ import annotations

from pathlib import PurePosixPath

# Languages with a tree-sitter grammar wired up in chunking.treesitter.
SYNTACTIC_LANGUAGES = frozenset(
    {"python", "javascript", "typescript", "tsx", "csharp", "java", "go", "rust"}
)

EXTENSIONS: dict[str, str] = {
    # syntactic
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".cs": "csharp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    # line-window fallback: other code
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".ps1": "powershell",
    ".sql": "sql",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "css",
    ".vue": "vue",
    ".svelte": "svelte",
    # docs & config
    ".md": "markdown",
    ".mdx": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".ini": "ini",
    ".cfg": "ini",
    ".xml": "xml",
    ".csproj": "xml",
    ".gradle": "gradle",
}

FILENAMES: dict[str, str] = {
    "dockerfile": "dockerfile",
    "makefile": "make",
    "readme": "text",
    "license": "text",
    "changelog": "text",
}


def detect_language(path: str) -> str | None:
    """Language for a repo-relative path, or ``None`` if the file type is not indexed."""
    p = PurePosixPath(path)
    if p.name.lower() in FILENAMES:
        return FILENAMES[p.name.lower()]
    return EXTENSIONS.get(p.suffix.lower())
