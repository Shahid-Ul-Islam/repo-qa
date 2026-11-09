from pathlib import Path

from conftest import CountingEmbedder
from repo_qa.embeddings.mock import MockEmbedder
from repo_qa.indexing.indexer import Indexer
from repo_qa.store.repository import Store


def test_first_index(store: Store, embedder: CountingEmbedder, sample_repo: Path) -> None:
    repo, stats = Indexer(store, embedder).index(sample_repo)
    assert stats.files_added == 4 and stats.files_scanned == 4
    assert stats.chunks_written > 4
    assert stats.embeddings_computed == embedder.documents_embedded > 0
    assert repo.embed_model == embedder.name and repo.embed_dim == 384
    assert repo.indexed_at is not None

    st = store.status(repo.id)
    assert st is not None
    assert st.file_count == 4
    assert st.embedding_count == stats.embeddings_computed
    assert st.languages == {"python": 2, "markdown": 1, "typescript": 1}
    assert store.get_file(repo.id, "src/shop/cart.py").line_count == 16

    symbols = {c.symbol for c in store.chunks_for_path(repo.id, "src/shop/cart.py")}
    assert {"Cart", "Cart.add_item", "Cart.total_price"} <= symbols


def test_reindex_unchanged_does_no_work(
    store: Store, embedder: CountingEmbedder, sample_repo: Path
) -> None:
    indexer = Indexer(store, embedder)
    indexer.index(sample_repo)
    before = embedder.documents_embedded
    _, stats = indexer.index(sample_repo)
    assert stats.files_unchanged == 4
    assert stats.files_added == stats.files_updated == stats.chunks_written == 0
    assert stats.embeddings_computed == 0 and stats.embeddings_reused > 0
    assert embedder.documents_embedded == before


def test_editing_one_function_reembeds_only_that_chunk(
    store: Store, embedder: CountingEmbedder, sample_repo: Path
) -> None:
    indexer = Indexer(store, embedder)
    repo, _ = indexer.index(sample_repo)
    cart = sample_repo / "src/shop/cart.py"
    cart.write_text(
        cart.read_text().replace(
            "self.items.append((sku, quantity))", "self.items.append((sku, 2))"
        ),
        encoding="utf-8",
        newline="\n",
    )
    before = embedder.documents_embedded
    _, stats = indexer.index(sample_repo)
    assert stats.files_updated == 1 and stats.files_unchanged == 3
    assert stats.embeddings_computed == 1  # only Cart.add_item's content changed
    assert embedder.documents_embedded - before == 1
    add_item = [
        c for c in store.chunks_for_path(repo.id, "src/shop/cart.py") if c.symbol == "Cart.add_item"
    ]
    assert "(sku, 2)" in add_item[0].text


def test_deleted_file_is_pruned_with_its_vectors(
    store: Store, embedder: CountingEmbedder, sample_repo: Path
) -> None:
    indexer = Indexer(store, embedder)
    repo, _ = indexer.index(sample_repo)
    vectors_before = store.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    (sample_repo / "src/shop/payment.py").unlink()
    _, stats = indexer.index(sample_repo)
    assert stats.files_removed == 1
    assert store.get_file(repo.id, "src/shop/payment.py") is None
    assert store.chunks_for_path(repo.id, "src/shop/payment.py") == []
    assert store.fts_search(repo.id, "refund", 10) == []
    vectors_after = store.conn.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert vectors_after < vectors_before


def test_switching_model_embeds_everything_again(
    store: Store, embedder: CountingEmbedder, sample_repo: Path
) -> None:
    repo, first = Indexer(store, embedder).index(sample_repo)
    other = CountingEmbedder(dim=64)
    repo, stats = Indexer(store, other).index(sample_repo)
    assert stats.files_unchanged == 4
    assert stats.embeddings_computed == first.embeddings_computed
    assert repo.embed_model == "mock-hash-64" and repo.embed_dim == 64
    ids, matrix = store.load_vectors(repo.id, other.name)
    assert matrix.shape[1] == 64 and len(ids) == first.chunks_written


def test_force_rechunks_but_reuses_vectors(
    store: Store, embedder: CountingEmbedder, sample_repo: Path
) -> None:
    indexer = Indexer(store, embedder)
    indexer.index(sample_repo)
    _, stats = indexer.index(sample_repo, force=True)
    assert stats.files_updated == 4 and stats.embeddings_computed == 0


def test_crlf_and_bom_do_not_shift_lines(store: Store, tmp_path: Path) -> None:
    root = tmp_path / "r"
    root.mkdir()
    (root / "a.py").write_bytes(b"\xef\xbb\xbfimport os\r\n\r\ndef f():\r\n    return 1\r\n")
    repo, _ = Indexer(store, MockEmbedder()).index(root)
    f = next(c for c in store.chunks_for_path(repo.id, "a.py") if c.symbol == "f")
    assert (f.start_line, f.end_line) == (3, 4)
    assert "\r" not in f.text
