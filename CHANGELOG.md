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
- Syntactic chunking via tree-sitter for Python, JavaScript, TypeScript/TSX, C#, Java, Go and Rust: one chunk per function/method (qualified `Type.method`, Go receivers, Rust impls) including leading comments/decorators, type-header and module gap chunks so every line is indexed once, windowing for oversized units, and line-window fallback on failure. Pins `tree-sitter==0.25.2` (0.26.0 crashes on Windows).
- Pluggable embedding providers: `EmbeddingProvider` protocol, local fastembed provider (lazy-loaded `BAAI/bge-small-en-v1.5`, passage/query modes) and a deterministic feature-hashing mock for tests and CI.
- Incremental indexer: skips files whose git blob hash is unchanged, re-chunks changed files, embeds only content hashes with no vector for the current model (batched, resumable), prunes deleted files and orphaned vectors, and records HEAD commit and model; `force` re-chunks everything.
- Hybrid retrieval: cached in-memory cosine search, sanitised FTS5 BM25 keyword search (stopwords dropped, identifiers split), reciprocal rank fusion (k=60), include/exclude path globs, and a guard against querying with a different embedding model than the index.
- Context pack builder: merges overlapping/adjacent hits per file into stitched regions (best score, union of symbols), then greedily packs regions by score into a token budget (~4 chars/token), truncating the top region rather than returning an empty pack.
