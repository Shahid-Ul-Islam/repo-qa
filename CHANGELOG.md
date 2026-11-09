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
