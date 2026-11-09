"""SQLite connection management and schema migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA_V1 = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS repos (
    id          INTEGER PRIMARY KEY,
    root_path   TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    head_commit TEXT,
    embed_model TEXT,
    embed_dim   INTEGER,
    indexed_at  TEXT
);

CREATE TABLE IF NOT EXISTS files (
    id         INTEGER PRIMARY KEY,
    repo_id    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path       TEXT NOT NULL,
    language   TEXT NOT NULL,
    size       INTEGER NOT NULL,
    line_count INTEGER NOT NULL,
    blob_hash  TEXT NOT NULL,
    UNIQUE (repo_id, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id           INTEGER PRIMARY KEY,
    repo_id      INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    file_id      INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    path         TEXT NOT NULL,
    language     TEXT NOT NULL,
    kind         TEXT NOT NULL,
    symbol       TEXT,
    start_line   INTEGER NOT NULL,
    end_line     INTEGER NOT NULL,
    text         TEXT NOT NULL,
    terms        TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL,
    blob_hash    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_repo_path ON chunks(repo_id, path);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);

-- Content-addressed: identical chunk text is embedded once per model.
CREATE TABLE IF NOT EXISTS embeddings (
    content_hash TEXT NOT NULL,
    model        TEXT NOT NULL,
    dim          INTEGER NOT NULL,
    vector       BLOB NOT NULL,
    PRIMARY KEY (content_hash, model)
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, symbol, path, terms,
    content='chunks', content_rowid='id',
    tokenize='porter unicode61'  -- stemming: "refunds" matches "refund"
);

CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, symbol, path, terms)
    VALUES (new.id, new.text, coalesce(new.symbol, ''), new.path, new.terms);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, symbol, path, terms)
    VALUES ('delete', old.id, old.text, coalesce(old.symbol, ''), old.path, old.terms);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, symbol, path, terms)
    VALUES ('delete', old.id, old.text, coalesce(old.symbol, ''), old.path, old.terms);
    INSERT INTO chunks_fts(rowid, text, symbol, path, terms)
    VALUES (new.id, new.text, coalesce(new.symbol, ''), new.path, new.terms);
END;

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    repo_id    INTEGER NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id             INTEGER PRIMARY KEY,
    session_id     TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role           TEXT NOT NULL,
    content        TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

_MIGRATIONS: dict[int, str] = {1: _SCHEMA_V1}


class SchemaError(RuntimeError):
    pass


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (creating if needed) an index database and bring its schema up to date."""
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    _require_fts5(conn)
    migrate(conn)
    return conn


def _require_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp._fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE temp._fts5_probe")
    except sqlite3.OperationalError as exc:  # pragma: no cover - depends on the platform build
        raise SchemaError("This Python's SQLite build lacks FTS5, which repo-qa requires.") from exc


def schema_version(conn: sqlite3.Connection) -> int:
    has_meta = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='meta'"
    ).fetchone()
    if not has_meta:
        return 0
    row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
    return int(row["value"]) if row else 0


def migrate(conn: sqlite3.Connection) -> None:
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise SchemaError(
            f"Index schema v{current} is newer than this repo-qa (v{SCHEMA_VERSION}); upgrade."
        )
    for version in range(current + 1, SCHEMA_VERSION + 1):
        with conn:
            conn.executescript(_MIGRATIONS[version])
            conn.execute(
                "INSERT INTO meta(key, value) VALUES ('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (str(version),),
            )
