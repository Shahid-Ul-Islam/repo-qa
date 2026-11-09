"""Per-session conversation memory, persisted in the index database."""

from __future__ import annotations

from collections.abc import Sequence

from repo_qa.models import Answer, Message
from repo_qa.store.repository import Store


class SessionMemory:
    def __init__(self, store: Store, turns: int = 4) -> None:
        self.store = store
        self.turns = turns

    def start(self, repo_id: int) -> str:
        return self.store.create_session(repo_id)

    def belongs_to(self, session_id: str, repo_id: int) -> bool:
        return self.store.session_repo(session_id) == repo_id

    def history(self, session_id: str | None) -> list[Message]:
        """The last ``turns`` question/answer pairs, oldest first."""
        if not session_id or self.turns == 0:
            return []
        return self.store.get_messages(session_id, limit=self.turns * 2)

    def transcript(self, session_id: str) -> list[Message]:
        return self.store.get_messages(session_id)

    def record(self, session_id: str, answer: Answer) -> None:
        self.store.add_message(session_id, "user", answer.question)
        self.store.add_message(session_id, "assistant", answer.text, answer.citations)


def retrieval_query(question: str, history: Sequence[Message]) -> str:
    """Follow-ups like "and where is it called?" carry little signal on their own, so the
    previous question is appended to the retrieval query (not to the prompt question)."""
    previous = next((m.content for m in reversed(history) if m.role == "user"), None)
    return f"{question}\n{previous}" if previous else question
