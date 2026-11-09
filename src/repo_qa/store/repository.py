"""Typed data access over the index database."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from repo_qa.models import (
    Citation,
    CodeChunk,
    FileRecord,
    Message,
    RepoRecord,
    RepoStatus,
    StoredChunk,
)
from repo_qa.store.db import connect
from repo_qa.text import identifier_terms

_CHUNK_COLS = (
    "id, repo_id, path, language, kind, symbol, start_line, end_line, text, content_hash, blob_hash"
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _row_to_chunk(row: sqlite3.Row) -> StoredChunk:
    return StoredChunk(**{k: row[k] for k in row.keys()})  # noqa: SIM118 - Row is not a dict


def _row_to_repo(row: sqlite3.Row) -> RepoRecord:
    data = {k: row[k] for k in row.keys()}  # noqa: SIM118
    if data.get("indexed_at"):
        data["indexed_at"] = datetime.fromisoformat(data["indexed_at"])
    return RepoRecord(**data)


class Store:
    """All SQL lives here; the rest of the code works with Pydantic models."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @classmethod
    def open(cls, path: Path | str) -> Store:
        return cls(connect(path))

    def close(self) -> None:
        self.conn.close()

    # ------------------------------------------------------------------ repos
    def upsert_repo(self, root_path: str, name: str) -> RepoRecord:
        with self.conn:
            self.conn.execute(
                "INSERT INTO repos(root_path, name) VALUES (?, ?) "
                "ON CONFLICT(root_path) DO UPDATE SET name = excluded.name",
                (root_path, name),
            )
        repo = self.get_repo_by_path(root_path)
        assert repo is not None
        return repo

    def get_repo(self, repo_id: int) -> RepoRecord | None:
        row = self.conn.execute("SELECT * FROM repos WHERE id = ?", (repo_id,)).fetchone()
        return _row_to_repo(row) if row else None

    def get_repo_by_path(self, root_path: str) -> RepoRecord | None:
        row = self.conn.execute("SELECT * FROM repos WHERE root_path = ?", (root_path,)).fetchone()
        return _row_to_repo(row) if row else None

    def list_repos(self) -> list[RepoRecord]:
        rows = self.conn.execute("SELECT * FROM repos ORDER BY name, id").fetchall()
        return [_row_to_repo(r) for r in rows]

    def mark_indexed(
        self, repo_id: int, *, head_commit: str | None, embed_model: str, embed_dim: int
    ) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE repos SET head_commit = ?, embed_model = ?, embed_dim = ?, indexed_at = ? "
                "WHERE id = ?",
                (head_commit, embed_model, embed_dim, _now(), repo_id),
            )

    def delete_repo(self, repo_id: int) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM repos WHERE id = ?", (repo_id,))

    # ------------------------------------------------------------------ files
    def get_files(self, repo_id: int) -> dict[str, FileRecord]:
        rows = self.conn.execute(
            "SELECT path, language, size, line_count, blob_hash FROM files WHERE repo_id = ?",
            (repo_id,),
        ).fetchall()
        return {r["path"]: FileRecord(**dict(r)) for r in rows}

    def get_file(self, repo_id: int, path: str) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT path, language, size, line_count, blob_hash FROM files "
            "WHERE repo_id = ? AND path = ?",
            (repo_id, path),
        ).fetchone()
        return FileRecord(**dict(row)) if row else None

    def write_file(
        self, repo_id: int, record: FileRecord, chunks: Sequence[tuple[CodeChunk, str]]
    ) -> None:
        """Insert or replace a file and all its chunks atomically.

        ``chunks`` pairs each chunk with its content hash.
        """
        with self.conn:
            self.conn.execute(
                "DELETE FROM files WHERE repo_id = ? AND path = ?", (repo_id, record.path)
            )
            cur = self.conn.execute(
                "INSERT INTO files(repo_id, path, language, size, line_count, blob_hash) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    repo_id,
                    record.path,
                    record.language,
                    record.size,
                    record.line_count,
                    record.blob_hash,
                ),
            )
            file_id = cur.lastrowid
            self.conn.executemany(
                "INSERT INTO chunks(repo_id, file_id, path, language, kind, symbol, start_line, "
                "end_line, text, terms, content_hash, blob_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        repo_id,
                        file_id,
                        record.path,
                        c.language,
                        c.kind.value,
                        c.symbol,
                        c.start_line,
                        c.end_line,
                        c.text,
                        identifier_terms(f"{c.symbol or ''}\n{c.text}"),
                        content_hash,
                        record.blob_hash,
                    )
                    for c, content_hash in chunks
                ],
            )

    def delete_files(self, repo_id: int, paths: Iterable[str]) -> int:
        paths = list(paths)
        with self.conn:
            self.conn.executemany(
                "DELETE FROM files WHERE repo_id = ? AND path = ?", [(repo_id, p) for p in paths]
            )
        return len(paths)

    # ----------------------------------------------------------------- chunks
    def get_chunk(self, chunk_id: int) -> StoredChunk | None:
        row = self.conn.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE id = ?", (chunk_id,)
        ).fetchone()
        return _row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Sequence[int]) -> dict[int, StoredChunk]:
        out: dict[int, StoredChunk] = {}
        ids = list(dict.fromkeys(chunk_ids))
        for i in range(0, len(ids), 500):
            batch = ids[i : i + 500]
            marks = ",".join("?" * len(batch))
            for row in self.conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks WHERE id IN ({marks})", batch
            ):
                out[row["id"]] = _row_to_chunk(row)
        return out

    def chunks_for_path(self, repo_id: int, path: str) -> list[StoredChunk]:
        rows = self.conn.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE repo_id = ? AND path = ? ORDER BY start_line",
            (repo_id, path),
        ).fetchall()
        return [_row_to_chunk(r) for r in rows]

    def chunk_hashes(self, repo_id: int) -> set[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT content_hash FROM chunks WHERE repo_id = ?", (repo_id,)
        )
        return {r[0] for r in rows}

    def chunks_by_hash(self, hashes: Iterable[str]) -> dict[str, StoredChunk]:
        """One representative chunk per content hash (identical hash = identical input)."""
        out: dict[str, StoredChunk] = {}
        hs = list(hashes)
        for i in range(0, len(hs), 500):
            batch = hs[i : i + 500]
            marks = ",".join("?" * len(batch))
            for row in self.conn.execute(
                f"SELECT {_CHUNK_COLS} FROM chunks WHERE id IN ("
                f"SELECT min(id) FROM chunks WHERE content_hash IN ({marks}) "
                "GROUP BY content_hash)",
                batch,
            ):
                out[row["content_hash"]] = _row_to_chunk(row)
        return out

    # ------------------------------------------------------------- embeddings
    def existing_embeddings(self, hashes: Iterable[str], model: str) -> set[str]:
        found: set[str] = set()
        hs = list(hashes)
        for i in range(0, len(hs), 500):
            batch = hs[i : i + 500]
            marks = ",".join("?" * len(batch))
            rows = self.conn.execute(
                "SELECT content_hash FROM embeddings "
                f"WHERE model = ? AND content_hash IN ({marks})",
                [model, *batch],
            )
            found.update(r[0] for r in rows)
        return found

    def put_embeddings(self, model: str, items: Sequence[tuple[str, np.ndarray]]) -> None:
        with self.conn:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings(content_hash, model, dim, vector) "
                "VALUES (?, ?, ?, ?)",
                [
                    (h, model, int(v.shape[0]), np.asarray(v, dtype=np.float32).tobytes())
                    for h, v in items
                ],
            )

    def load_vectors(self, repo_id: int, model: str) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(chunk_ids, matrix)`` for every embedded chunk of a repo."""
        rows = self.conn.execute(
            "SELECT c.id, e.dim, e.vector FROM chunks c "
            "JOIN embeddings e ON e.content_hash = c.content_hash AND e.model = ? "
            "WHERE c.repo_id = ? ORDER BY c.id",
            (model, repo_id),
        ).fetchall()
        if not rows:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float32)
        ids = np.fromiter((r[0] for r in rows), dtype=np.int64, count=len(rows))
        matrix = np.vstack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        return ids, matrix

    def gc_embeddings(self) -> int:
        """Delete vectors no chunk (in any repo) refers to any more."""
        with self.conn:
            cur = self.conn.execute(
                "DELETE FROM embeddings WHERE content_hash NOT IN "
                "(SELECT DISTINCT content_hash FROM chunks)"
            )
        return cur.rowcount

    # -------------------------------------------------------------- full text
    def fts_search(self, repo_id: int, match: str, limit: int) -> list[tuple[int, float]]:
        """Run an FTS5 MATCH; returns ``(chunk_id, score)`` with higher = better."""
        rows = self.conn.execute(
            "SELECT c.id, bm25(chunks_fts, 1.0, 4.0, 2.0, 1.0) AS rank "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? AND c.repo_id = ? "
            "ORDER BY rank LIMIT ?",
            (match, repo_id, limit),
        ).fetchall()
        # bm25() is "lower is better" and negative; flip the sign for readability.
        return [(r[0], -float(r[1])) for r in rows]

    # ----------------------------------------------------------------- status
    def status(self, repo_id: int) -> RepoStatus | None:
        repo = self.get_repo(repo_id)
        if repo is None:
            return None
        file_count = self.conn.execute(
            "SELECT count(*) FROM files WHERE repo_id = ?", (repo_id,)
        ).fetchone()[0]
        chunk_count = self.conn.execute(
            "SELECT count(*) FROM chunks WHERE repo_id = ?", (repo_id,)
        ).fetchone()[0]
        embedding_count = self.conn.execute(
            "SELECT count(DISTINCT c.content_hash) FROM chunks c JOIN embeddings e "
            "ON e.content_hash = c.content_hash AND e.model = ? WHERE c.repo_id = ?",
            (repo.embed_model or "", repo_id),
        ).fetchone()[0]
        languages = {
            r[0]: r[1]
            for r in self.conn.execute(
                "SELECT language, count(*) FROM files WHERE repo_id = ? "
                "GROUP BY language ORDER BY count(*) DESC, language",
                (repo_id,),
            )
        }
        return RepoStatus(
            repo=repo,
            file_count=file_count,
            chunk_count=chunk_count,
            embedding_count=embedding_count,
            languages=languages,
        )

    # --------------------------------------------------------------- sessions
    def create_session(self, repo_id: int) -> str:
        session_id = uuid.uuid4().hex
        with self.conn:
            self.conn.execute(
                "INSERT INTO sessions(id, repo_id, created_at) VALUES (?, ?, ?)",
                (session_id, repo_id, _now()),
            )
        return session_id

    def session_repo(self, session_id: str) -> int | None:
        row = self.conn.execute(
            "SELECT repo_id FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row[0] if row else None

    def add_message(
        self, session_id: str, role: str, content: str, citations: Sequence[Citation] = ()
    ) -> None:
        payload = json.dumps([c.model_dump(mode="json") for c in citations])
        with self.conn:
            self.conn.execute(
                "INSERT INTO messages(session_id, role, content, citations_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, payload, _now()),
            )

    def get_messages(self, session_id: str, limit: int | None = None) -> list[Message]:
        """Messages in chronological order; ``limit`` keeps only the most recent ones."""
        sql = (
            "SELECT role, content, citations_json, created_at FROM ("
            "SELECT id, role, content, citations_json, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id"
        )
        params = (session_id, -1 if limit is None else limit)  # LIMIT -1 = no limit
        return [
            Message(
                role=r["role"],
                content=r["content"],
                citations=[Citation(**c) for c in json.loads(r["citations_json"])],
                created_at=datetime.fromisoformat(r["created_at"]),
            )
            for r in self.conn.execute(sql, params)
        ]
