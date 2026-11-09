"""OpenAI via the official SDK's Responses API (``pip install repo-qa[openai]``)."""

from __future__ import annotations

import os
from typing import Any

from repo_qa.answering.providers.base import LLMRequest, ProviderError

DEFAULT_MODEL = "gpt-5"


class OpenAIProvider:
    def __init__(
        self, model: str | None = None, *, max_output_tokens: int = 16000, client: Any = None
    ) -> None:
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self.name = f"openai:{self.model}"
        self.max_output_tokens = max_output_tokens
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:
                raise ProviderError(
                    "The OpenAI provider needs `pip install repo-qa[openai]`."
                ) from exc
            if not os.environ.get("OPENAI_API_KEY"):
                raise ProviderError("Set OPENAI_API_KEY to use the OpenAI provider.")
            self._client = openai.OpenAI()
        return self._client

    def complete(self, request: LLMRequest) -> str:
        client = self._get_client()  # raises ProviderError with an install hint if missing
        import openai

        try:
            response = client.responses.create(
                model=self.model,
                instructions=request.system,
                input=[{"role": m.role, "content": m.content} for m in request.messages],
                max_output_tokens=self.max_output_tokens,
            )
        except openai.APIConnectionError as exc:
            raise ProviderError(f"Could not reach the OpenAI API: {exc}") from exc
        except openai.APIStatusError as exc:
            raise ProviderError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
        text = (response.output_text or "").strip()
        if not text:
            raise ProviderError("Empty response from OpenAI.")
        return text
