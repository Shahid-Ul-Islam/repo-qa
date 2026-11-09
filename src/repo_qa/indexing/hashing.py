"""Content hashing for incremental indexing."""

from __future__ import annotations

import hashlib


def git_blob_hash(data: bytes) -> str:
    """The SHA-1 git would assign to ``data`` as a blob (``git hash-object``).

    Computed in-process so indexing does not spawn one git subprocess per file. It equals
    git's own hash whenever the working-tree bytes match the committed blob (i.e. no
    line-ending or clean/smudge filter conversion).
    """
    h = hashlib.sha1(usedforsecurity=False)
    h.update(b"blob %d\0" % len(data))
    h.update(data)
    return h.hexdigest()


def content_hash(text: str) -> str:
    """Stable identity of a chunk's text; embeddings are cached under this key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
