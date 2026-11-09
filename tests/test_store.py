from pathlib import Path

import numpy as np
import pytest

from repo_qa.models import ChunkKind, Citation, CitationStatus, CodeChunk, FileRecord
from repo_qa.store.db import SCHEMA_VERSION, SchemaError, connect, schema_version
from repo_qa.store.repository import Store
from repo_qa.text import identifier_terms, split_identifier, tokenize


@pytest.fixture
def store(tmp_path: Path) -> Store:
    s = Store.open(tmp_path / "idx.db")
    yield s
    s.close()


def _chunk(start: int, end: int, text: str, symbol: str | None = None) -> CodeChunk:
    return CodeChunk(
        language="python",
        kind=ChunkKind.FUNCTION,
        symbol=symbol,
        start_line=start,
        end_line=end,
        text=text,
    )


def _file(path: str, blob: str = "b1", lines: int = 20) -> FileRecord:
    return FileRecord(path=path, language="python", size=100, line_count=lines, blob_hash=blob)


def test_split_identifier() -> None:
    assert split_identifier("parseHTTPRequest_v2") == ["parse", "http", "request", "v", "2"]
    assert split_identifier("simple") == ["simple"]
    assert identifier_terms("def load_vectors(): getUserName") == "load vectors get user name"
    assert "rrf" in tokenize("rrf_fuse") and "fuse" in tokenize("rrf_fuse")


def test_schema_created_and_versioned(tmp_path: Path) -> None:
    conn = connect(tmp_path / "a.db")
    assert schema_version(conn) == SCHEMA_VERSION
    conn.close()
    conn = connect(tmp_path / "a.db")  # idempotent reopen
    assert schema_version(conn) == SCHEMA_VERSION
    conn.execute("UPDATE meta SET value = '999' WHERE key = 'schema_version'")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaError):
        connect(tmp_path / "a.db")


def test_write_file_and_fts_follow_cascade(store: Store) -> None:
    repo = store.upsert_repo("/r", "r")
    store.write_file(
        repo.id,
        _file("a.py"),
        [(_chunk(1, 5, "def parseHttpRequest(): pass", "parseHttpRequest"), "h1")],
    )
    hits = store.fts_search(repo.id, '"http"', 10)  # matches via split identifier terms
    assert len(hits) == 1
    chunk = store.get_chunk(hits[0][0])
    assert chunk is not None and chunk.location == "a.py:1-5"

    # Rewriting the file replaces its chunks, and the FTS index follows.
    store.write_file(repo.id, _file("a.py", "b2"), [(_chunk(1, 3, "def other(): pass"), "h2")])
    assert store.fts_search(repo.id, '"http"', 10) == []
    assert len(store.fts_search(repo.id, "other", 10)) == 1

    store.delete_files(repo.id, ["a.py"])
    assert store.fts_search(repo.id, "other", 10) == []
    assert store.get_files(repo.id) == {}


def test_fts_scoped_to_repo(store: Store) -> None:
    r1 = store.upsert_repo("/r1", "r1")
    r2 = store.upsert_repo("/r2", "r2")
    store.write_file(r1.id, _file("a.py"), [(_chunk(1, 2, "alpha beta"), "h1")])
    store.write_file(r2.id, _file("a.py"), [(_chunk(1, 2, "alpha gamma"), "h2")])
    assert len(store.fts_search(r1.id, "alpha", 10)) == 1
    assert store.fts_search(r1.id, "gamma", 10) == []


def test_embeddings_roundtrip_and_gc(store: Store) -> None:
    repo = store.upsert_repo("/r", "r")
    store.write_file(repo.id, _file("a.py"), [(_chunk(1, 2, "x"), "h1"), (_chunk(3, 4, "x"), "h1")])
    v = np.array([0.6, 0.8], dtype=np.float32)
    store.put_embeddings("m", [("h1", v), ("orphan", v)])
    assert store.existing_embeddings(["h1", "h2", "orphan"], "m") == {"h1", "orphan"}
    ids, matrix = store.load_vectors(repo.id, "m")
    assert len(ids) == 2 and matrix.shape == (2, 2)  # both chunks share one vector
    np.testing.assert_allclose(matrix[0], v)
    assert store.gc_embeddings() == 1
    assert store.existing_embeddings(["orphan"], "m") == set()


def test_status(store: Store) -> None:
    repo = store.upsert_repo("/r", "r")
    store.write_file(repo.id, _file("a.py"), [(_chunk(1, 2, "x"), "h1")])
    store.put_embeddings("m", [("h1", np.ones(2, dtype=np.float32))])
    store.mark_indexed(repo.id, head_commit="abc", embed_model="m", embed_dim=2)
    st = store.status(repo.id)
    assert st is not None
    assert (st.file_count, st.chunk_count, st.embedding_count) == (1, 1, 1)
    assert st.languages == {"python": 1}
    assert st.repo.indexed_at is not None and st.repo.head_commit == "abc"
    assert store.status(999) is None


def test_sessions_and_messages(store: Store) -> None:
    repo = store.upsert_repo("/r", "r")
    sid = store.create_session(repo.id)
    assert store.session_repo(sid) == repo.id
    cit = Citation(
        raw="a.py:1-2", path="a.py", start_line=1, end_line=2, status=CitationStatus.VERIFIED
    )
    for i in range(5):
        store.add_message(sid, "user", f"q{i}")
        store.add_message(sid, "assistant", f"a{i}", [cit])
    msgs = store.get_messages(sid)
    assert len(msgs) == 10 and msgs[0].content == "q0"
    recent = store.get_messages(sid, limit=4)
    assert [m.content for m in recent] == ["q3", "a3", "q4", "a4"]
    assert recent[-1].citations[0].status is CitationStatus.VERIFIED
