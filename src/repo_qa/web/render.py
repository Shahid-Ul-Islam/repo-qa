"""Safe HTML rendering of answers: escape first, then add citation links and code spans."""

from __future__ import annotations

import html
import re
from urllib.parse import urlencode

from markupsafe import Markup

from repo_qa.answering.citations import CITATION_RE, INVALID_MARK
from repo_qa.models import Answer, Citation

_CODE = re.compile(r"`([^`\n]+)`")


def chunk_url(repo_id: int, path: str, start: int, end: int) -> str:
    query = urlencode({"path": path, "start": start, "end": end})
    return f"/repos/{repo_id}/view?{query}"


def _citation_link(repo_id: int, raw: str, c: Citation) -> str:
    title = html.escape(c.reason or "verified against the index", quote=True)
    url = html.escape(chunk_url(repo_id, c.path, c.start_line, c.end_line), quote=True)
    return (
        f'<a class="cite {c.status.value}" href="{url}" hx-get="{url}" hx-target="#viewer" '
        f'title="{title}">{raw}</a>'
    )


def render_answer(answer: Answer, repo_id: int) -> Markup:
    """Answer text → HTML. All model output is escaped before any markup is added."""
    by_raw = {c.raw: c for c in answer.citations}
    text = html.escape(answer.text, quote=False)

    def link(m: re.Match[str]) -> str:
        c = by_raw.get(m.group(0))
        return _citation_link(repo_id, m.group(0), c) if c else m.group(0)

    text = CITATION_RE.sub(link, text)
    text = text.replace(
        html.escape(INVALID_MARK), f'<span class="invalid">{html.escape(INVALID_MARK)}</span>'
    )
    text = _CODE.sub(r"<code>\1</code>", text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return Markup("".join(f"<p>{p.replace(chr(10), '<br>')}</p>" for p in paragraphs))
