# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.1.0] - 2026-11-09

First release: index a local repository and ask questions answered from its code, with
every `path:start-end` citation verified against the index. Works through the CLI or web UI,
offline by default.

### Added

#### Indexing
- File walker that honours `.gitignore` via `git ls-files`, with a nested-`.gitignore`
  fallback for directories that aren't git repos. It skips binaries, vendored directories,
  lockfiles, minified files and files over a size limit, and reports why each file was
  skipped.
- Syntactic chunking with tree-sitter for Python, JavaScript, TypeScript/TSX, C#, Java, Go
  and Rust:
  - One chunk per function or method, including its leading comments and decorators, with
    a qualified symbol (`Type.method`, Go receivers, Rust `impl` types).
  - Lines outside functions go into type-header and module chunks, so every line is
    indexed exactly once.
  - Oversized units are windowed and keep their symbol.
- Line-window chunking (60 lines, 10 overlap) for other languages, and heading-based
  Markdown chunking with breadcrumb symbols.
- Incremental re-indexing:
  - Files with an unchanged git blob hash are skipped.
  - Embeddings are keyed by a SHA-256 hash of each chunk's content, so only new content is
    embedded.
  - Deleted files and orphaned vectors are removed.
  - The HEAD commit and embedding model are recorded; `--force` re-chunks every file.
- Line endings are normalised to LF, so hashes and line numbers match across OSes.

#### Storage
- Central SQLite index (`~/.repo-qa/index.db`) with a versioned schema, holding:
  - files and chunks,
  - an FTS5 keyword index kept in sync by triggers, with Porter stemming and
    identifiers split into words (`parseHttpRequest` → `parse http request`),
  - embeddings keyed by content hash,
  - conversation sessions.

#### Embeddings
- Pluggable `EmbeddingProvider` interface, with a local fastembed provider
  (`BAAI/bge-small-en-v1.5`, no API key) and a deterministic mock for tests and CI.

#### Retrieval
- Hybrid search: in-memory cosine similarity plus FTS5 BM25 keyword search (user input
  sanitised), fused with reciprocal rank fusion (k=60), with include/exclude path globs.
- Refuses to search when the index was built with a different embedding model than the
  one configured.
- Context pack builder: merges overlapping hits per file, then fills a token budget in
  score order.

#### Answering
- LLM providers:
  - Anthropic: defaults to `claude-opus-5-5` at medium effort, with server-side refusal
    fallbacks and refusal handling.
  - OpenAI: Responses API.
  - A deterministic offline mock.
  - A clear `pip install repo-qa[...]` hint when a provider's SDK isn't installed.
- Grounded prompt with line-numbered excerpts and a `NOT_FOUND` sentinel.
- Citation parsing and verification:
  - `verified`: exists in the index and overlaps the context the model was given.
  - `ungrounded`: exists but wasn't in the context; kept, but flagged.
  - `invalid`: unknown path or impossible line range; removed and flagged inline.
- "Not found in the repo" when retrieval confidence is low, without calling the model.
- Per-session conversation memory, with the previous question added to retrieval for
  follow-ups.

#### Interfaces
- CLI: `repo-qa index`, `search`, `ask`, `eval`, `serve`, `status`, with `--json` output
  and repo selection by id, name, path or current directory.
- Web UI (FastAPI + Jinja2 + vendored htmx 2.0.4, no build step):
  - repo picker and conversational ask form with path filters,
  - clickable citations that open the cited lines highlighted,
  - a side panel of retrieved chunks with their RRF, vector and keyword scores,
  - an index status page.
- `repo-qa serve` binds to 127.0.0.1 by default and warns when exposed beyond localhost.

#### Evaluation
- `repo-qa eval` runs JSONL eval sets and reports hit-rate@k, MRR@k and substring
  hit-rate. It also reports citation precision with `--answers`, or by default when a real
  LLM provider is configured. `--min-hit-rate` makes it usable as a CI gate.
- Ships `evals/repo_qa_self.jsonl`, 22 questions about repo-qa's own code, which the test
  suite runs offline. Measured with the mock embedder:

  | Scope | hit-rate@5 | MRR@5 | Floor enforced |
  |---|---|---|---|
  | Source code only (tests, evals, docs excluded) | 1.000 | 0.720 | 0.90 / 0.60 |
  | Whole repo | 0.773 | 0.513 | 0.70 / 0.45 |

#### Project
- Exact dependency pins; `tree-sitter` is held at 0.25.2 because 0.26.0 corrupts the heap
  on Windows.
- GitHub Actions CI on Ubuntu and Windows (ruff check, ruff format, pytest), fully
  offline using the mock providers.
- README with an architecture diagram, quickstart, design decisions and roadmap; MIT
  license.

[Unreleased]: https://github.com/Shahid-Ul-Islam/repo-qa/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Shahid-Ul-Islam/repo-qa/releases/tag/v0.1.0
