"""FastAPI + Jinja2 + HTMX web UI (no JavaScript build step)."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from repo_qa.answering.providers.base import ProviderError
from repo_qa.config import Settings
from repo_qa.models import RepoRecord
from repo_qa.retrieval.search import IndexMismatchError
from repo_qa.runtime import Runtime
from repo_qa.web.render import chunk_url, render_answer

HERE = Path(__file__).parent
VIEW_PADDING = 6  # extra lines shown around a cited range


def _split_globs(value: str) -> list[str]:
    return [g.strip() for g in value.replace(",", " ").split() if g.strip()]


def create_app(runtime: Runtime | None = None, settings: Settings | None = None) -> FastAPI:
    rt = runtime or Runtime.create(settings or Settings.from_env())
    app = FastAPI(title="repo-qa", docs_url=None, redoc_url=None)
    app.state.runtime = rt
    # One SQLite connection is shared; requests use it one at a time.
    lock = threading.Lock()
    templates = Jinja2Templates(directory=HERE / "templates")
    templates.env.globals["chunk_url"] = chunk_url
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")

    @contextmanager
    def using() -> Iterator[Runtime]:
        with lock:
            yield rt

    def get_repo(r: Runtime, repo_id: int) -> RepoRecord:
        repo = r.store.get_repo(repo_id)
        if repo is None:
            raise HTTPException(status_code=404, detail="Unknown repository")
        return repo

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request, repo: int | None = None) -> HTMLResponse:
        with using() as r:
            repos = r.store.list_repos()
        selected = next((x for x in repos if x.id == repo), repos[0] if repos else None)
        return templates.TemplateResponse(
            request,
            "index.html",
            {"repos": repos, "selected": selected, "provider": rt.provider.name},
        )

    @app.post("/ask", response_class=HTMLResponse)
    def ask(
        request: Request,
        repo_id: Annotated[int, Form()],
        question: Annotated[str, Form()],
        session_id: Annotated[str, Form()] = "",
        include: Annotated[str, Form()] = "",
        exclude: Annotated[str, Form()] = "",
    ) -> HTMLResponse:
        question = question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="Question is empty")
        error = None
        answer = None
        with using() as r:
            repo = get_repo(r, repo_id)
            if not session_id or r.store.session_repo(session_id) != repo.id:
                session_id = r.engine().memory.start(repo.id)
            try:
                answer = r.engine().ask(
                    repo,
                    question,
                    session_id=session_id,
                    include=_split_globs(include),
                    exclude=_split_globs(exclude),
                )
            except (ProviderError, IndexMismatchError) as exc:
                error = str(exc)
        return templates.TemplateResponse(
            request,
            "_turn.html",
            {
                "repo": repo,
                "question": question,
                "answer": answer,
                "answer_html": render_answer(answer, repo.id) if answer else None,
                "error": error,
                "session_id": session_id,
            },
        )

    @app.get("/repos/{repo_id}/view", response_class=HTMLResponse)
    def view(
        request: Request,
        repo_id: int,
        path: Annotated[str, Query()],
        start: Annotated[int, Query(ge=1)],
        end: Annotated[int, Query(ge=1)],
    ) -> HTMLResponse:
        """Show indexed lines around ``start..end``, stitched from stored chunks so the view
        always matches what citations were verified against."""
        with using() as r:
            repo = get_repo(r, repo_id)
            record = r.store.get_file(repo.id, path)
            if record is None:
                raise HTTPException(status_code=404, detail="File not in index")
            chunks = r.store.chunks_for_path(repo.id, path)
        end = max(start, min(end, record.line_count))
        lo, hi = max(1, start - VIEW_PADDING), min(record.line_count, end + VIEW_PADDING)
        lines: dict[int, str] = {}
        symbols: list[str] = []
        for c in chunks:
            if c.end_line < lo or c.start_line > hi:
                continue
            for n, line in enumerate(c.text.split("\n"), start=c.start_line):
                lines.setdefault(n, line)
            if c.symbol and c.start_line <= end and c.end_line >= start:
                symbols.append(c.symbol)
        rows = [(n, lines.get(n, ""), start <= n <= end) for n in range(lo, hi + 1)]
        return templates.TemplateResponse(
            request,
            "_viewer.html",
            {
                "repo": repo,
                "path": path,
                "start": start,
                "end": end,
                "rows": rows,
                "language": record.language,
                "symbols": list(dict.fromkeys(symbols)),
            },
        )

    @app.get("/status", response_class=HTMLResponse)
    def status(request: Request) -> HTMLResponse:
        with using() as r:
            statuses = [s for s in (r.store.status(x.id) for x in r.store.list_repos()) if s]
        return templates.TemplateResponse(
            request,
            "status.html",
            {
                "statuses": statuses,
                "db_path": rt.settings.db_path,
                "embedder": rt.embedder.name,
                "provider": rt.provider.name,
            },
        )

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app
