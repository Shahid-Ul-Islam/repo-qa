from repo_qa.answering.citations import INVALID_MARK
from repo_qa.answering.memory import retrieval_query
from repo_qa.answering.providers.base import LLMRequest
from repo_qa.answering.providers.mock import MockProvider
from repo_qa.answering.qa import NOT_FOUND_MESSAGE, QAEngine, retrieval_confidence
from repo_qa.models import CitationStatus, Message
from repo_qa.retrieval.search import Searcher


class ScriptedProvider:
    """Returns a fixed reply and records every request."""

    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> str:
        self.requests.append(request)
        return self.reply


class SpySearcher(Searcher):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.queries: list[str] = []

    def search(self, repo, query, **kwargs):
        self.queries.append(query)
        return super().search(repo, query, **kwargs)


def _engine(indexed, provider, **kw):
    store, embedder, repo = indexed
    searcher = SpySearcher(store, embedder)
    return QAEngine(store, searcher, provider, **kw), searcher, repo


def test_answer_with_verified_citations(indexed) -> None:
    engine, _, repo = _engine(indexed, MockProvider())
    answer = engine.ask(repo, "How do we charge a credit card?")
    assert answer.found
    assert answer.citations and all(c.status is CitationStatus.VERIFIED for c in answer.citations)
    assert answer.citations[0].path == "src/shop/payment.py"
    assert answer.context and answer.hits
    assert answer.confidence_reason == "keyword match"


def test_low_confidence_skips_the_model(indexed) -> None:
    provider = ScriptedProvider("should not be called")
    engine, _, repo = _engine(indexed, provider)
    answer = engine.ask(repo, "What is the capital of France?")
    assert not answer.found and answer.text == NOT_FOUND_MESSAGE
    assert answer.citations == [] and provider.requests == []
    assert answer.confidence_reason == "low retrieval confidence"


def test_model_not_found_sentinel(indexed) -> None:
    engine, _, repo = _engine(indexed, ScriptedProvider("NOT_FOUND"))
    answer = engine.ask(repo, "How are refunds handled?")
    assert not answer.found and answer.text == NOT_FOUND_MESSAGE
    assert answer.context  # retrieval happened; the model declined


def test_hallucinated_citations_are_flagged(indexed) -> None:
    reply = (
        "Refunds call refund_payment (src/shop/payment.py:8-10), "
        "logged in src/shop/audit.py:1-5, see also src/shop/cart.py:1-2."
    )
    engine, _, repo = _engine(indexed, ScriptedProvider(reply))
    answer = engine.ask(repo, "How are payments refunded?", include=["src/shop/payment.py"])
    by_path = {c.path: c.status for c in answer.citations}
    assert by_path["src/shop/payment.py"] is CitationStatus.VERIFIED
    assert by_path["src/shop/cart.py"] is CitationStatus.UNGROUNDED  # real lines, not in context
    assert "src/shop/audit.py" not in by_path  # invalid citations are dropped...
    assert f"src/shop/audit.py:1-5 {INVALID_MARK}" in answer.text  # ...and flagged inline


def test_session_memory_and_follow_up_retrieval(indexed) -> None:
    _store, _, repo = indexed
    provider = ScriptedProvider("It is in src/shop/cart.py:10-12.")
    engine, searcher, _ = _engine(indexed, provider)
    sid = engine.memory.start(repo.id)
    engine.ask(repo, "How is an item added to the cart?", session_id=sid)
    engine.ask(repo, "And how is the total computed?", session_id=sid)

    # The follow-up retrieves with the previous question as extra signal...
    assert (
        searcher.queries[1] == "And how is the total computed?\nHow is an item added to the cart?"
    )
    # ...and the model sees the earlier turn as history.
    second = provider.requests[1]
    assert [m.role for m in second.messages] == ["user", "assistant", "user"]
    assert second.messages[0].content == "How is an item added to the cart?"
    transcript = engine.memory.transcript(sid)
    assert len(transcript) == 4 and transcript[1].citations[0].path == "src/shop/cart.py"


def test_history_window_is_bounded(indexed) -> None:
    _store, _, repo = indexed
    provider = ScriptedProvider("See src/shop/cart.py:10-12.")
    engine, _, _ = _engine(indexed, provider, history_turns=1)
    sid = engine.memory.start(repo.id)
    for q in ("cart add item", "cart total price", "cart items list"):
        engine.ask(repo, q, session_id=sid)
    assert len(provider.requests[-1].messages) == 3  # one prior turn + the new question


def test_session_from_another_repo_is_ignored(indexed, sample_repo) -> None:
    store, _, repo = indexed
    other = store.upsert_repo("/elsewhere", "elsewhere")
    foreign = store.create_session(other.id)
    engine, _, _ = _engine(indexed, MockProvider())
    answer = engine.ask(repo, "charge credit card", session_id=foreign)
    assert answer.session_id is None
    assert store.get_messages(foreign) == []


def test_retrieval_query_helper() -> None:
    hist = [Message(role="user", content="q1"), Message(role="assistant", content="a1")]
    assert retrieval_query("q2", hist) == "q2\nq1"
    assert retrieval_query("q2", []) == "q2"


def test_retrieval_confidence_rules(indexed) -> None:
    store, embedder, repo = indexed
    hits = Searcher(store, embedder).search(repo, "credit card", top_k=3)
    assert retrieval_confidence(hits, 0.99) == "keyword match"
    assert retrieval_confidence([], 0.0) is None
    no_kw = [h.model_copy(update={"keyword_rank": None}) for h in hits]
    assert retrieval_confidence(no_kw, 0.99) is None
    assert retrieval_confidence(no_kw, 0.01).startswith("vector similarity")
