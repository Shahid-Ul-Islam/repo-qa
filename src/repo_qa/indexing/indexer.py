"""Incremental indexing: walk → chunk changed files → embed new content → prune."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from repo_qa.chunking.base import chunk_file, normalize_text
from repo_qa.chunking.lines import split_lines
from repo_qa.embeddings.base import EmbeddingProvider, embedding_input
from repo_qa.indexing.hashing import content_hash, git_blob_hash
from repo_qa.indexing.walker import SourceFile, git_head, walk
from repo_qa.models import FileRecord, IndexStats, RepoRecord
from repo_qa.store.repository import Store

log = logging.getLogger(__name__)

Progress = Callable[[str], None]
EMBED_BATCH = 256


class Indexer:
    def __init__(
        self, store: Store, embedder: EmbeddingProvider, *, max_file_bytes: int = 1_000_000
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.max_file_bytes = max_file_bytes

    def index(
        self, root: Path, *, force: bool = False, progress: Progress | None = None
    ) -> tuple[RepoRecord, IndexStats]:
        """Bring the index for ``root`` up to date and return the repo and run statistics.

        Files whose git blob hash is unchanged are skipped entirely; changed files are
        re-chunked; only chunk contents never embedded before (for this model) are sent to
        the embedder. ``force`` re-chunks every file (embeddings are still reused).
        """
        started = time.perf_counter()
        say = progress or (lambda _msg: None)
        root = root.resolve()
        repo = self.store.upsert_repo(str(root), root.name)
        stats = IndexStats()

        walked = walk(root, max_bytes=self.max_file_bytes)
        say(f"found {len(walked.files)} indexable files")
        existing = self.store.get_files(repo.id)
        seen: set[str] = set()

        for source in walked.files:
            stats.files_scanned += 1
            seen.add(source.path)
            data = source.abs_path.read_bytes()
            blob = git_blob_hash(data)
            previous = existing.get(source.path)
            if (
                not force
                and previous is not None
                and previous.blob_hash == blob
                and previous.language == source.language
            ):
                stats.files_unchanged += 1
                continue
            stats.chunks_written += self._write_file(repo.id, source, data, blob)
            if previous is None:
                stats.files_added += 1
            else:
                stats.files_updated += 1

        removed = set(existing) - seen
        stats.files_removed = self.store.delete_files(repo.id, sorted(removed))
        if removed:
            say(f"removed {len(removed)} deleted files")

        computed, reused = self._embed_missing(repo.id, say)
        stats.embeddings_computed, stats.embeddings_reused = computed, reused
        self.store.gc_embeddings()
        self.store.mark_indexed(
            repo.id,
            head_commit=git_head(root),
            embed_model=self.embedder.name,
            embed_dim=self.embedder.dim,
        )
        stats.duration_s = round(time.perf_counter() - started, 3)
        updated = self.store.get_repo(repo.id)
        assert updated is not None
        return updated, stats

    def _write_file(self, repo_id: int, source: SourceFile, data: bytes, blob: str) -> int:
        text = normalize_text(data.decode("utf-8", errors="replace").removeprefix("﻿"))
        chunks = chunk_file(text, source.language)
        record = FileRecord(
            path=source.path,
            language=source.language,
            size=source.size,
            line_count=len(split_lines(text)),
            blob_hash=blob,
        )
        pairs = [(c, content_hash(embedding_input(source.path, c))) for c in chunks]
        self.store.write_file(repo_id, record, pairs)
        return len(pairs)

    def _embed_missing(self, repo_id: int, say: Progress) -> tuple[int, int]:
        model = self.embedder.name
        hashes = self.store.chunk_hashes(repo_id)
        missing = sorted(hashes - self.store.existing_embeddings(hashes, model))
        if missing:
            say(f"embedding {len(missing)} new chunks with {model}")
        for i in range(0, len(missing), EMBED_BATCH):
            batch = missing[i : i + EMBED_BATCH]
            chunks = self.store.chunks_by_hash(batch)
            inputs = [embedding_input(chunks[h].path, chunks[h]) for h in batch]
            vectors = self.embedder.embed_documents(inputs)
            # Persist per batch so an interrupted run keeps its progress.
            self.store.put_embeddings(model, list(zip(batch, vectors, strict=True)))
        return len(missing), len(hashes) - len(missing)
