from types import SimpleNamespace

import pytest

from repo_qa.answering.citations import parse_citations
from repo_qa.answering.prompts import (
    NOT_FOUND,
    SYSTEM_PROMPT,
    build_request,
    is_not_found,
    render_item,
)
from repo_qa.answering.providers.anthropic import AnthropicProvider
from repo_qa.answering.providers.base import LLMProvider, ProviderError, get_provider
from repo_qa.answering.providers.mock import MockProvider
from repo_qa.answering.providers.openai import OpenAIProvider
from repo_qa.config import Settings
from repo_qa.models import ContextItem, ContextPack, Message


def _item(path: str, start: int, end: int, sym: str | None = None) -> ContextItem:
    text = "\n".join(f"code {i}" for i in range(start, end + 1))
    return ContextItem(
        path=path,
        language="python",
        start_line=start,
        end_line=end,
        text=text,
        score=1.0,
        symbols=[sym] if sym else [],
    )


PACK = ContextPack(
    items=[_item("src/a.py", 9, 11, "f"), _item("src/b.py", 1, 2), _item("c.py", 5, 5)],
    token_count=50,
    token_budget=1000,
)


def test_render_item_numbers_lines() -> None:
    rendered = render_item(_item("src/a.py", 9, 11, "f"))
    assert rendered.splitlines()[0] == "[src/a.py:9-11] (f)"
    assert " 9| code 9" in rendered and "11| code 11" in rendered


def test_build_request_puts_context_in_last_user_turn() -> None:
    history = [
        Message(role="assistant", content="orphan"),
        Message(role="user", content="q1"),
        Message(role="assistant", content="a1"),
    ]
    req = build_request("where is f?", PACK, history)
    assert req.system == SYSTEM_PROMPT
    assert [m.role for m in req.messages] == ["user", "assistant", "user"]
    last = req.messages[-1].content
    assert last.startswith("<context>") and last.endswith("Question: where is f?")
    assert "[src/b.py:1-2]" in last


def test_system_prompt_is_stable() -> None:
    assert (
        build_request("a", PACK).system
        == build_request("b", ContextPack(items=[], token_count=0, token_budget=1)).system
    )


def test_mock_answers_with_parseable_citations() -> None:
    answer = MockProvider().complete(build_request("q", PACK))
    assert [c.location for c in parse_citations(answer)] == [
        "src/a.py:9-11",
        "src/b.py:1-2",
        "c.py:5-5",
    ]
    assert "`f`" in answer
    assert MockProvider().complete(build_request("q", PACK)) == answer  # deterministic


def test_mock_not_found_without_context() -> None:
    empty = ContextPack(items=[], token_count=0, token_budget=100)
    assert MockProvider().complete(build_request("q", empty)) == NOT_FOUND


@pytest.mark.parametrize(
    ("text", "expected"),
    [("NOT_FOUND", True), (" not_found. ", True), ("NOT_FOUND because x", False)],
)
def test_is_not_found(text: str, expected: bool) -> None:
    assert is_not_found(text) is expected


def test_factory_and_protocol() -> None:
    assert isinstance(get_provider(Settings(llm="mock")), MockProvider)
    a = get_provider(Settings(llm="anthropic"))
    assert isinstance(a, AnthropicProvider) and a.model == "claude-opus-5-5"
    o = get_provider(Settings(llm="openai", llm_model="some-model"))
    assert isinstance(o, OpenAIProvider) and o.name == "openai:some-model"
    for p in (MockProvider(), a, o):
        assert isinstance(p, LLMProvider)


class _FakeAnthropicMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def _fake_anthropic(response):
    messages = _FakeAnthropicMessages(response)
    return SimpleNamespace(beta=SimpleNamespace(messages=messages)), messages


def test_anthropic_request_shape_and_text_extraction() -> None:
    response = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text="See src/a.py:9-11."),
        ],
    )
    client, messages = _fake_anthropic(response)
    out = AnthropicProvider(client=client).complete(build_request("q", PACK))
    assert out == "See src/a.py:9-11."
    kw = messages.kwargs
    assert kw["model"] == "claude-opus-5-5" and kw["system"] == SYSTEM_PROMPT
    assert kw["messages"][-1]["role"] == "user"
    assert kw["fallbacks"] == "default" and kw["betas"] == ["server-side-fallback-2026-07-01"]
    assert kw["output_config"] == {"effort": "medium"}


def test_anthropic_refusal_raises() -> None:
    response = SimpleNamespace(
        stop_reason="refusal", content=[], stop_details=SimpleNamespace(category="cyber")
    )
    client, _ = _fake_anthropic(response)
    with pytest.raises(ProviderError, match="cyber"):
        AnthropicProvider(client=client).complete(build_request("q", PACK))


def test_openai_request_shape() -> None:
    captured = {}

    def create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_text="Answer src/b.py:1-2")

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    out = OpenAIProvider(model="m", client=client).complete(build_request("q", PACK))
    assert out == "Answer src/b.py:1-2"
    assert captured["model"] == "m" and captured["instructions"] == SYSTEM_PROMPT


def test_openai_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider().complete(build_request("q", PACK))


@pytest.mark.parametrize("module", ["anthropic", "openai"])
def test_missing_sdk_gives_install_hint(monkeypatch, module) -> None:
    import sys

    monkeypatch.setitem(sys.modules, module, None)  # simulate the extra not being installed
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    provider = AnthropicProvider() if module == "anthropic" else OpenAIProvider()
    with pytest.raises(ProviderError, match=rf"repo-qa\[{module}\]"):
        provider.complete(build_request("q", PACK))
