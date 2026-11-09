"""Shared fixtures: everything runs offline with the mock embedder and a temp database."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from repo_qa.embeddings.mock import MockEmbedder
from repo_qa.indexing.indexer import Indexer
from repo_qa.store.repository import Store


class CountingEmbedder(MockEmbedder):
    """Mock embedder that records how many documents it was asked to embed."""

    def __init__(self, dim: int = 384) -> None:
        super().__init__(dim)
        self.documents_embedded = 0

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        self.documents_embedded += len(texts)
        return super().embed_documents(texts)


SAMPLE_REPO = {
    "src/shop/cart.py": '''\
"""Shopping cart."""


class Cart:
    """Holds line items."""

    def __init__(self):
        self.items = []

    def add_item(self, sku, quantity=1):
        """Add a product to the cart."""
        self.items.append((sku, quantity))

    def total_price(self, prices):
        """Sum price * quantity for every item."""
        return sum(prices[sku] * qty for sku, qty in self.items)
''',
    "src/shop/payment.py": '''\
def charge_credit_card(card_number, amount):
    """Charge a credit card through the payment gateway."""
    if amount <= 0:
        raise ValueError("amount must be positive")
    return {"card": card_number[-4:], "charged": amount}


def refund_payment(transaction_id):
    """Refund a previous payment transaction."""
    return {"refunded": transaction_id}
''',
    "web/router.ts": """\
export function parseRoute(url: string): string[] {
  return url.split("/").filter(Boolean);
}

export class Router {
  navigate(path: string): void {
    window.history.pushState({}, "", path);
  }
}
""",
    "docs/guide.md": """\
# Guide

Intro to the shop.

## Checkout

Customers pay with a credit card at checkout.

## Shipping

Orders ship within two days.
""",
}


def write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    s = Store.open(tmp_path / "index.db")
    yield s
    s.close()


@pytest.fixture
def embedder() -> CountingEmbedder:
    return CountingEmbedder()


@pytest.fixture
def sample_repo(tmp_path: Path) -> Path:
    root = tmp_path / "shop"
    write_tree(root, SAMPLE_REPO)
    return root


@pytest.fixture
def indexed(store: Store, embedder: CountingEmbedder, sample_repo: Path):
    """``(store, embedder, repo)`` with the sample repo already indexed."""
    repo, _ = Indexer(store, embedder).index(sample_repo)
    return store, embedder, repo
