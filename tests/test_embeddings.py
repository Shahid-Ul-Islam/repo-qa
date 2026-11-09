import os

import numpy as np
import pytest

from repo_qa.config import Settings
from repo_qa.embeddings import EmbeddingProvider, embedding_input, get_embedder
from repo_qa.embeddings.fastembed_provider import FastEmbedProvider
from repo_qa.embeddings.mock import MockEmbedder
from repo_qa.models import ChunkKind, CodeChunk


def test_mock_is_deterministic_and_normalized() -> None:
    a, b = MockEmbedder(), MockEmbedder()
    texts = ["def parse_config(path): ...", "class HttpClient: pass"]
    va, vb = a.embed_documents(texts), b.embed_documents(texts)
    np.testing.assert_array_equal(va, vb)
    assert va.shape == (2, 384) and va.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(va, axis=1), 1.0, rtol=1e-5)


def test_mock_similarity_tracks_lexical_overlap() -> None:
    e = MockEmbedder()
    docs = e.embed_documents(
        ["def load_config(path): read the yaml config file", "def render_chart(data): draw bars"]
    )
    q = e.embed_query("how is the config file loaded?")
    sims = docs @ q
    assert sims[0] > sims[1]
    # identifier parts match plain words
    assert e.embed_query("config loader") @ e.embed_query("ConfigLoader") > 0.5


def test_mock_handles_empty_input() -> None:
    e = MockEmbedder(dim=16)
    assert e.embed_documents([]).shape == (0, 16)
    assert not np.isnan(e.embed_query("")).any()


def test_providers_satisfy_protocol() -> None:
    assert isinstance(MockEmbedder(), EmbeddingProvider)
    assert isinstance(FastEmbedProvider(), EmbeddingProvider)


def test_factory() -> None:
    assert isinstance(get_embedder(Settings(embedder="mock")), MockEmbedder)
    fe = get_embedder(Settings(embedder="fastembed"))
    assert isinstance(fe, FastEmbedProvider) and fe.dim == 384  # metadata only, no download


def test_embedding_input_includes_location() -> None:
    chunk = CodeChunk(
        language="python",
        kind=ChunkKind.FUNCTION,
        symbol="f",
        start_line=1,
        end_line=1,
        text="def f(): ...",
    )
    assert embedding_input("src/a.py", chunk) == "src/a.py f\ndef f(): ..."


@pytest.mark.skipif(not os.environ.get("REPO_QA_TEST_FASTEMBED"), reason="downloads an ONNX model")
def test_fastembed_smoke() -> None:
    e = FastEmbedProvider()
    docs = e.embed_documents(["def add(a, b): return a + b", "README: install with pip"])
    assert docs.shape == (2, e.dim)
    assert (docs @ e.embed_query("function that adds two numbers")).argmax() == 0
