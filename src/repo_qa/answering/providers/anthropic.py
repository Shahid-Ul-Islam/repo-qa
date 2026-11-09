"""Claude via the official Anthropic SDK (``pip install repo-qa[anthropic]``)."""

from __future__ import annotations

from typing import Any

from repo_qa.answering.providers.base import LLMRequest, ProviderError

DEFAULT_MODEL = "claude-opus-5-5"
# Server-side refusal fallback: on a safety decline the API re-runs the request on a
# suitable fallback model inside the same call, so a refusal does not end the answer.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicProvider:
    def __init__(
        self,
        model: str | None = None,
        *,
        max_tokens: int = 16000,
        effort: str = "medium",
        client: Any = None,
    ) -> None:
        self.model = model or DEFAULT_MODEL
        self.name = f"anthropic:{self.model}"
        self.max_tokens = max_tokens
        self.effort = effort
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ProviderError(
                    "The Anthropic provider needs `pip install repo-qa[anthropic]`."
                ) from exc
            # Resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profiles.
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, request: LLMRequest) -> str:
        import anthropic

        try:
            response = self._get_client().beta.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=request.system,
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                output_config={"effort": self.effort},
                betas=[FALLBACK_BETA],
                fallbacks="default",
            )
        except anthropic.APIConnectionError as exc:
            raise ProviderError(f"Could not reach the Anthropic API: {exc}") from exc
        except anthropic.APIStatusError as exc:
            raise ProviderError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise ProviderError(f"The model declined to answer (category: {category}).")
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        if not text:
            raise ProviderError(f"Empty response (stop_reason={response.stop_reason}).")
        return text
