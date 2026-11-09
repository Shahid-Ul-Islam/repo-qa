# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Project scaffold: pyproject (hatchling, pinned dependencies), ruff config, MIT license,
  `.gitignore`, `.env.example`, README stub.
- GitHub Actions CI (Ubuntu + Windows): ruff check, ruff format, pytest.
- `.gitattributes` normalising line endings to LF so hashes and line numbers match across OSes.
- Pydantic domain models (chunks, files, repos, search hits, context packs, citations, answers, eval) and `Settings` read from `REPO_QA_*` env vars.
- SQLite store: versioned schema, FTS5 external-content table kept in sync by triggers (with split-identifier `terms` column), content-addressed embeddings, sessions/messages, and a typed `Store` repository.
- File walker honouring `.gitignore` (via `git ls-files`, with a `pathspec` fallback for non-git dirs) that skips binaries, vendored dirs, lockfiles, minified files and oversized files; in-process git blob hashing and SHA-256 chunk content hashing.
- Line-window chunker (60 lines, 10 overlap, blank edges trimmed) and Markdown chunker splitting on headings with breadcrumb symbols, fence-aware, windowing oversized sections.
