import re

import pytest
from fastapi.testclient import TestClient

from repo_qa.answering.providers.base import LLMRequest, ProviderError
from repo_qa.answering.providers.mock import MockProvider
from repo_qa.config import Settings
from repo_qa.runtime import Runtime
from repo_qa.web.app import create_app


class ScriptedProvider:
    name = "scripted"

    def __init__(self, reply: str | Exception) -> None:
        self.reply = reply

    def complete(self, request: LLMRequest) -> str:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def _client(indexed, tmp_path, provider=None) -> TestClient:
    store, embedder, _ = indexed
    rt = Runtime(
        settings=Settings(db_path=tmp_path / "index.db", embedder="mock"),
        store=store,
        embedder=embedder,
        provider=provider or MockProvider(),
    )
    return TestClient(create_app(runtime=rt))


@pytest.fixture
def client(indexed, tmp_path) -> TestClient:
    return _client(indexed, tmp_path)


def _session_id(html: str) -> str:
    m = re.search(r'id="session_id" value="([0-9a-f]+)" hx-swap-oob', html)
    assert m, html
    return m.group(1)


def test_index_page_lists_repos(client, indexed) -> None:
    _, _, repo = indexed
    r = client.get("/")
    assert r.status_code == 200
    assert f'<option value="{repo.id}" selected>shop</option>' in r.text
    assert 'hx-post="/ask"' in r.text and "/static/htmx.min.js" in r.text


def test_index_page_without_repos(store, embedder, tmp_path) -> None:
    rt = Runtime(Settings(db_path=tmp_path / "x.db"), store, embedder, MockProvider())
    r = TestClient(create_app(runtime=rt)).get("/")
    assert "No repositories are indexed yet" in r.text


def test_ask_renders_answer_citations_and_hits(client, indexed) -> None:
    _, _, repo = indexed
    r = client.post(
        "/ask", data={"repo_id": repo.id, "question": "How do we charge a credit card?"}
    )
    assert r.status_code == 200
    html = r.text
    assert 'class="cite verified"' in html
    assert f"/repos/{repo.id}/view?path=src%2Fshop%2Fpayment.py&amp;start=1&amp;end=5" in html
    assert 'id="hits" hx-swap-oob="true"' in html and "rrf " in html
    assert "<code>charge_credit_card</code>" in html  # symbol shown in the side panel
    assert _session_id(html)


def test_follow_up_uses_the_same_session(client, indexed) -> None:
    store, _, repo = indexed
    first = client.post("/ask", data={"repo_id": repo.id, "question": "cart add item"})
    sid = _session_id(first.text)
    second = client.post(
        "/ask", data={"repo_id": repo.id, "question": "and the total price?", "session_id": sid}
    )
    assert _session_id(second.text) == sid
    assert len(store.get_messages(sid)) == 4


def test_unknown_session_starts_a_new_one(client, indexed) -> None:
    _, _, repo = indexed
    r = client.post("/ask", data={"repo_id": repo.id, "question": "cart", "session_id": "deadbeef"})
    assert _session_id(r.text) != "deadbeef"


def test_not_found_answer(client, indexed) -> None:
    _, _, repo = indexed
    r = client.post("/ask", data={"repo_id": repo.id, "question": "capital of France?"})
    assert "Not found in the repo" in r.text and "low retrieval confidence" in r.text


def test_filters_are_applied(client, indexed) -> None:
    _, _, repo = indexed
    r = client.post(
        "/ask",
        data={"repo_id": repo.id, "question": "credit card", "include": "docs/*"},
    )
    locs = re.findall(r'class="loc">([^<]+)<', r.text)
    assert locs and all(loc.startswith("docs/guide.md") for loc in locs)


def test_model_output_is_escaped(indexed, tmp_path) -> None:
    _, _, repo = indexed
    client = _client(
        indexed, tmp_path, ScriptedProvider("<script>alert(1)</script> see src/shop/cart.py:10-12")
    )
    r = client.post("/ask", data={"repo_id": repo.id, "question": "cart add item"})
    assert "<script>alert(1)</script>" not in r.text
    assert "&lt;script&gt;" in r.text
    assert 'class="cite verified"' in r.text


def test_invalid_citation_is_flagged(indexed, tmp_path) -> None:
    _, _, repo = indexed
    client = _client(indexed, tmp_path, ScriptedProvider("It lives in src/shop/ghost.py:1-3."))
    r = client.post("/ask", data={"repo_id": repo.id, "question": "cart add item"})
    assert '<span class="invalid">' in r.text
    assert "No verifiable citations" in r.text


def test_provider_error_is_shown(indexed, tmp_path) -> None:
    _, _, repo = indexed
    client = _client(indexed, tmp_path, ScriptedProvider(ProviderError("no API key")))
    r = client.post("/ask", data={"repo_id": repo.id, "question": "cart add item"})
    assert r.status_code == 200 and "no API key" in r.text


def test_ask_validation(client, indexed) -> None:
    _, _, repo = indexed
    assert client.post("/ask", data={"repo_id": repo.id, "question": "   "}).status_code == 422
    assert client.post("/ask", data={"repo_id": 999, "question": "x"}).status_code == 404


def test_chunk_viewer_highlights_range(client, indexed) -> None:
    _, _, repo = indexed
    r = client.get(
        f"/repos/{repo.id}/view", params={"path": "src/shop/cart.py", "start": 10, "end": 12}
    )
    assert r.status_code == 200
    highlighted = re.findall(r'<span class="row hl"><span class="ln">(\d+)</span>', r.text)
    assert highlighted == ["10", "11", "12"]
    assert '<span class="ln">4</span>' in r.text  # padding before the range
    assert "Cart.add_item" in r.text


def test_chunk_viewer_errors(client, indexed) -> None:
    _, _, repo = indexed
    assert (
        client.get(
            f"/repos/{repo.id}/view", params={"path": "nope.py", "start": 1, "end": 2}
        ).status_code
        == 404
    )
    assert (
        client.get("/repos/999/view", params={"path": "a.py", "start": 1, "end": 2}).status_code
        == 404
    )
    assert (
        client.get(
            f"/repos/{repo.id}/view", params={"path": "src/shop/cart.py", "start": 0, "end": 2}
        ).status_code
        == 422
    )


def test_status_page(client) -> None:
    r = client.get("/status")
    assert r.status_code == 200
    assert "shop" in r.text and "mock-hash-384" in r.text and "python 2" in r.text


def test_static_and_health(client) -> None:
    assert client.get("/static/htmx.min.js").status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}
