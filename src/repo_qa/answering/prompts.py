"""Prompt construction for grounded, cited answers."""

from __future__ import annotations

from collections.abc import Sequence

from repo_qa.answering.providers.base import ChatMessage, LLMRequest
from repo_qa.models import ContextItem, ContextPack, Message
from repo_qa.retrieval.context import item_header

NOT_FOUND = "NOT_FOUND"

# Kept byte-stable (no timestamps or per-request data) so providers can cache it.
SYSTEM_PROMPT = f"""\
You answer questions about a software repository using only the source excerpts provided \
in the <context> block of the latest message. Each excerpt is headed by its location as \
[path:start-end] and every code line is prefixed with its line number.

Rules:
- Base every statement on the provided excerpts. Do not rely on outside knowledge of the \
project, and do not guess about code you were not shown.
- Cite the lines that support each claim in the form path:start-end (for example \
src/app/main.py:12-30), using the exact paths and line numbers from the excerpts. Cite the \
narrowest range that supports the claim.
- If the excerpts do not contain enough information to answer, reply with exactly \
{NOT_FOUND} and nothing else.
- Be concise: lead with the direct answer, then supporting detail. Use Markdown; put code \
identifiers in backticks.
- Earlier turns are conversation history; their excerpts are no longer available, so only \
cite locations shown in the current <context> block."""


def render_item(item: ContextItem) -> str:
    width = len(str(item.end_line))
    numbered = "\n".join(
        f"{n:>{width}}| {line}"
        for n, line in enumerate(item.text.split("\n"), start=item.start_line)
    )
    return f"{item_header(item)}\n```{item.language}\n{numbered}\n```"


def render_context(pack: ContextPack) -> str:
    if not pack.items:
        return "<context>\n(no relevant code was found)\n</context>"
    body = "\n\n".join(render_item(i) for i in pack.items)
    return f"<context>\n{body}\n</context>"


def build_request(question: str, pack: ContextPack, history: Sequence[Message] = ()) -> LLMRequest:
    messages = [ChatMessage(role=m.role, content=m.content) for m in history]
    # Providers require the conversation to start with a user turn.
    while messages and messages[0].role != "user":
        messages.pop(0)
    messages.append(
        ChatMessage(role="user", content=f"{render_context(pack)}\n\nQuestion: {question}")
    )
    return LLMRequest(system=SYSTEM_PROMPT, messages=messages, pack=pack, question=question)


def is_not_found(text: str) -> bool:
    return text.strip().strip(".").upper() == NOT_FOUND
