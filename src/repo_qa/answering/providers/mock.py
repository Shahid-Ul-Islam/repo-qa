"""Offline, deterministic provider: answers by pointing at the best context regions.

It follows the same contract as a real model — cite as ``path:start-end``, reply with the
``NOT_FOUND`` sentinel when there is no usable context — so the whole answer pipeline
(citation parsing, verification, memory, UI) is exercised in tests without API keys.
"""

from __future__ import annotations

from repo_qa.answering.prompts import NOT_FOUND
from repo_qa.answering.providers.base import LLMRequest

MAX_CITED = 3


class MockProvider:
    name = "mock"

    def complete(self, request: LLMRequest) -> str:
        items = request.pack.items[:MAX_CITED]
        if not items:
            return NOT_FOUND
        lead = items[0]
        what = f"`{lead.symbols[0]}`" if lead.symbols else "the code"
        lines = [
            f"The most relevant code for this question is {what} in "
            f"{lead.path}:{lead.start_line}-{lead.end_line}."
        ]
        for item in items[1:]:
            label = f" (`{item.symbols[0]}`)" if item.symbols else ""
            lines.append(f"- Related{label}: {item.path}:{item.start_line}-{item.end_line}")
        return "\n".join(lines)
