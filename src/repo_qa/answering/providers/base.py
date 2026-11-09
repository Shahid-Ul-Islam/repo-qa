"""LLM provider protocol and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from repo_qa.models import ContextPack

if TYPE_CHECKING:
    from repo_qa.config import Settings


class ProviderError(RuntimeError):
    """The provider could not produce an answer (missing key, API error, refusal...)."""


@dataclass
class ChatMessage:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class LLMRequest:
    system: str
    messages: list[ChatMessage]
    # The structured context behind the prompt; real providers only need the rendered
    # prompt, but the mock answers from it directly.
    pack: ContextPack = field(
        default_factory=lambda: ContextPack(items=[], token_count=0, token_budget=0)
    )
    question: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, request: LLMRequest) -> str: ...


def get_provider(settings: Settings) -> LLMProvider:
    if settings.llm == "anthropic":
        from repo_qa.answering.providers.anthropic import AnthropicProvider

        return AnthropicProvider(model=settings.llm_model)
    if settings.llm == "openai":
        from repo_qa.answering.providers.openai import OpenAIProvider

        return OpenAIProvider(model=settings.llm_model)
    from repo_qa.answering.providers.mock import MockProvider

    return MockProvider()
