"""Command-line interface: ``repo-qa index|search|ask|eval|serve|status``."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from repo_qa.answering.providers.base import ProviderError
from repo_qa.config import EmbedderName, LLMName, Settings
from repo_qa.models import Answer, CitationStatus
from repo_qa.retrieval.search import IndexMismatchError
from repo_qa.runtime import RepoNotFoundError, Runtime

app = typer.Typer(
    help="Index a local repository and ask grounded, cited questions about it.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DbOpt = Annotated[
    Path | None, typer.Option("--db", help="Index database (default ~/.repo-qa/index.db).")
]
RepoOpt = Annotated[
    str | None, typer.Option("--repo", "-r", help="Repo id, name or path (default: current).")
]
TopKOpt = Annotated[int | None, typer.Option("--top-k", "-k", min=1, help="Results to use.")]
IncludeOpt = Annotated[
    list[str] | None, typer.Option("--include", "-i", help="Only paths matching this glob.")
]
ExcludeOpt = Annotated[
    list[str] | None, typer.Option("--exclude", "-x", help="Skip paths matching this glob.")
]
EmbedderOpt = Annotated[EmbedderName | None, typer.Option("--embedder", help="Embedding provider.")]
LLMOpt = Annotated[LLMName | None, typer.Option("--llm", help="Answering provider.")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output.")]


@app.callback()
def _setup() -> None:
    # Piped output on Windows uses a legacy code page; degrade symbols instead of crashing.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def _runtime(**overrides: object) -> Runtime:
    return Runtime.create(Settings.from_env(**overrides))


def _fail(message: str) -> typer.Exit:
    console.print(f"[red]error:[/red] {message}")
    return typer.Exit(code=1)


@app.command()
def index(
    path: Annotated[Path, typer.Argument(help="Repository root to index.")] = Path("."),
    force: Annotated[bool, typer.Option(help="Re-chunk every file.")] = False,
    db: DbOpt = None,
    embedder: EmbedderOpt = None,
) -> None:
    """Index (or incrementally re-index) a repository."""
    if not path.is_dir():
        raise _fail(f"{path} is not a directory")
    rt = _runtime(db_path=db, embedder=embedder)
    try:
        with console.status("Indexing…") as status:
            repo, stats = rt.indexer().index(path, force=force, progress=status.update)
    finally:
        rt.close()
    table = Table(title=f"Indexed {repo.name}", show_header=False)
    for label, value in [
        ("files scanned", stats.files_scanned),
        (
            "added / updated / unchanged / removed",
            f"{stats.files_added} / {stats.files_updated} / {stats.files_unchanged} / "
            f"{stats.files_removed}",
        ),
        ("chunks written", stats.chunks_written),
        (
            "embeddings computed / reused",
            f"{stats.embeddings_computed} / {stats.embeddings_reused}",
        ),
        ("model", repo.embed_model),
        ("time", f"{stats.duration_s:.1f}s"),
    ]:
        table.add_row(label, str(value))
    console.print(table)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search text.")],
    repo: RepoOpt = None,
    top_k: TopKOpt = None,
    include: IncludeOpt = None,
    exclude: ExcludeOpt = None,
    db: DbOpt = None,
    embedder: EmbedderOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Hybrid (vector + keyword) search; prints ranked chunks with scores."""
    rt = _runtime(db_path=db, embedder=embedder)
    try:
        record = rt.resolve_repo(repo)
        hits = rt.searcher.search(
            record,
            query,
            top_k=top_k or rt.settings.top_k,
            include=include or [],
            exclude=exclude or [],
        )
    except (RepoNotFoundError, IndexMismatchError) as exc:
        raise _fail(str(exc)) from exc
    finally:
        rt.close()
    if as_json:
        typer.echo(json.dumps([h.model_dump(mode="json") for h in hits], indent=2))
        return
    if not hits:
        console.print("No results.")
        return
    table = Table(title=f"{record.name}: {query}")
    for col in ("#", "location", "symbol", "rrf", "vector", "keyword"):
        table.add_column(col)
    for n, h in enumerate(hits, start=1):
        table.add_row(
            str(n),
            h.chunk.location,
            h.chunk.symbol or "",
            f"{h.score:.4f}",
            f"{h.vector_score:.3f} (#{h.vector_rank})" if h.vector_rank else "-",
            f"{h.keyword_score:.2f} (#{h.keyword_rank})" if h.keyword_rank else "-",
        )
    console.print(table)


def _print_answer(answer: Answer, show_context: bool) -> None:
    console.print(Markdown(answer.text))
    if answer.citations:
        console.print()
        for c in answer.citations:
            mark = (
                "[green]✓[/green]" if c.status is CitationStatus.VERIFIED else "[yellow]![/yellow]"
            )
            note = f" [dim]({c.reason})[/dim]" if c.reason else ""
            console.print(f"{mark} {c.location}{note}")
    if show_context:
        console.print("\n[bold]Context[/bold]")
        for item in answer.context:
            console.print(f"  {item.location} [dim]score {item.score:.4f}[/dim]")
    console.print(f"\n[dim]{answer.provider} · {answer.confidence_reason}[/dim]")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="Natural-language question.")],
    repo: RepoOpt = None,
    top_k: TopKOpt = None,
    include: IncludeOpt = None,
    exclude: ExcludeOpt = None,
    db: DbOpt = None,
    embedder: EmbedderOpt = None,
    llm: LLMOpt = None,
    show_context: Annotated[bool, typer.Option(help="List the context regions used.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Answer a question from the indexed code, with verified citations."""
    rt = _runtime(db_path=db, embedder=embedder, llm=llm)
    try:
        record = rt.resolve_repo(repo)
        answer = rt.engine(top_k=top_k).ask(
            record, question, include=include or [], exclude=exclude or []
        )
    except (RepoNotFoundError, IndexMismatchError, ProviderError) as exc:
        raise _fail(str(exc)) from exc
    finally:
        rt.close()
    if as_json:
        typer.echo(answer.model_dump_json(indent=2))
    else:
        _print_answer(answer, show_context)
    if not answer.found:
        raise typer.Exit(code=2)


@app.command("eval")
def eval_cmd(
    file: Annotated[Path, typer.Argument(help="JSONL eval set.", exists=True, dir_okay=False)],
    repo: RepoOpt = None,
    k: Annotated[int, typer.Option("--k", "-k", min=1, help="Cut-off for hit-rate/MRR.")] = 5,
    answers: Annotated[
        bool | None,
        typer.Option(
            "--answers/--no-answers",
            help="Also answer each question to score citation precision "
            "(default: only with a real LLM provider).",
        ),
    ] = None,
    min_hit_rate: Annotated[
        float | None, typer.Option(help="Exit 1 if hit-rate@k is below this.")
    ] = None,
    include: IncludeOpt = None,
    exclude: ExcludeOpt = None,
    db: DbOpt = None,
    embedder: EmbedderOpt = None,
    llm: LLMOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Evaluate retrieval (hit-rate@k, MRR) and, optionally, citation precision."""
    from repo_qa.evaluation.runner import EvalFileError, load_cases, run_eval

    rt = _runtime(db_path=db, embedder=embedder, llm=llm)
    try:
        cases = load_cases(file)
        record = rt.resolve_repo(repo)
        with_answers = answers if answers is not None else rt.settings.llm != "mock"
        engine = rt.engine() if with_answers else None
        with console.status("Evaluating…") as status:
            report = run_eval(
                cases,
                record,
                rt.searcher,
                k=k,
                engine=engine,
                include=include or [],
                exclude=exclude or [],
                progress=lambda n, total: status.update(f"Evaluating {n}/{total}…"),
            )
    except (EvalFileError, RepoNotFoundError, IndexMismatchError, ProviderError) as exc:
        raise _fail(str(exc)) from exc
    finally:
        rt.close()

    if as_json:
        typer.echo(report.model_dump_json(indent=2))
    else:
        table = Table(title=f"{file.name} on {record.name} (k={k})")
        for col in ("#", "question", "rank", "substr", "cit. prec."):
            table.add_column(col)
        for n, r in enumerate(report.cases, start=1):
            table.add_row(
                str(n),
                r.question,
                str(r.first_relevant_rank) if r.hit else "[red]miss[/red]",
                "-" if r.substring_hit is None else ("yes" if r.substring_hit else "no"),
                "-" if r.citation_precision is None else f"{r.citation_precision:.2f}",
            )
        console.print(table)
        console.print(
            f"hit-rate@{k}: [bold]{report.hit_rate:.3f}[/bold]   MRR@{k}: "
            f"[bold]{report.mrr:.3f}[/bold]"
        )
        if report.substring_hit_rate is not None:
            console.print(f"substring hit-rate@{k}: {report.substring_hit_rate:.3f}")
        if report.citation_precision is not None:
            console.print(f"citation precision: {report.citation_precision:.3f}")
    if min_hit_rate is not None and report.hit_rate < min_hit_rate:
        console.print(f"[red]hit-rate {report.hit_rate:.3f} below {min_hit_rate}[/red]")
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind (local-only by default).")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8000,
    db: DbOpt = None,
    embedder: EmbedderOpt = None,
    llm: LLMOpt = None,
) -> None:
    """Start the web UI."""
    import uvicorn

    from repo_qa.web.app import create_app

    settings = Settings.from_env(db_path=db, embedder=embedder, llm=llm)
    if host not in ("127.0.0.1", "localhost", "::1"):
        console.print(
            "[yellow]warning:[/yellow] the web UI has no authentication; anyone who can reach "
            f"{host}:{port} can read indexed code and spend your LLM quota."
        )
    console.print(f"repo-qa web UI on http://{host}:{port}  (index: {settings.db_path})")
    uvicorn.run(create_app(settings=settings), host=host, port=port, log_level="warning")


@app.command()
def status(repo: RepoOpt = None, db: DbOpt = None, as_json: JsonOpt = False) -> None:
    """Show indexed repositories and their index statistics."""
    rt = _runtime(db_path=db)
    try:
        records = [rt.resolve_repo(repo)] if repo else rt.store.list_repos()
        statuses = [rt.store.status(r.id) for r in records]
    except RepoNotFoundError as exc:
        raise _fail(str(exc)) from exc
    finally:
        rt.close()
    if as_json:
        typer.echo(json.dumps([s.model_dump(mode="json") for s in statuses if s], indent=2))
        return
    if not statuses:
        console.print("No repositories indexed yet. Run `repo-qa index <path>`.")
        return
    table = Table(title=f"Index: {rt.settings.db_path}")
    for col in (
        "id",
        "name",
        "path",
        "files",
        "chunks",
        "embedded",
        "model",
        "indexed at",
        "languages",
    ):
        table.add_column(col)
    for s in statuses:
        if s is None:
            continue
        r = s.repo
        langs = ", ".join(f"{k} {v}" for k, v in list(s.languages.items())[:5])
        table.add_row(
            str(r.id),
            r.name,
            r.root_path,
            str(s.file_count),
            str(s.chunk_count),
            str(s.embedding_count),
            r.embed_model or "-",
            r.indexed_at.strftime("%Y-%m-%d %H:%M") if r.indexed_at else "-",
            langs,
        )
    console.print(table)


if __name__ == "__main__":  # pragma: no cover
    app()
