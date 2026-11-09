"""Run a JSONL eval set against an indexed repo."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import ValidationError

from repo_qa.answering.qa import QAEngine
from repo_qa.evaluation.metrics import (
    citation_precision,
    first_relevant_rank,
    mean,
    reciprocal_rank,
    substring_hit,
)
from repo_qa.models import EvalCase, EvalCaseResult, EvalReport, RepoRecord
from repo_qa.retrieval.search import Searcher


class EvalFileError(ValueError):
    pass


def load_cases(path: Path) -> list[EvalCase]:
    """One JSON object per line; blank lines and ``//`` or ``#`` comments are ignored."""
    cases: list[EvalCase] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue
        try:
            cases.append(EvalCase.model_validate(json.loads(stripped)))
        except (json.JSONDecodeError, ValidationError) as exc:
            raise EvalFileError(f"{path}:{lineno}: {exc}") from exc
    if not cases:
        raise EvalFileError(f"{path}: no eval cases found")
    return cases


def run_eval(
    cases: list[EvalCase],
    repo: RepoRecord,
    searcher: Searcher,
    *,
    k: int = 5,
    engine: QAEngine | None = None,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    progress: Callable[[int, int], None] | None = None,
) -> EvalReport:
    """Retrieval metrics for every case; citation precision too when ``engine`` is given."""
    results: list[EvalCaseResult] = []
    for n, case in enumerate(cases, start=1):
        hits = searcher.search(repo, case.question, top_k=k, include=include, exclude=exclude)
        paths = [h.chunk.path for h in hits]
        rank = first_relevant_rank(paths, case.expected_paths)
        precision = None
        if engine is not None:
            answer = engine.ask(repo, case.question, include=include, exclude=exclude)
            precision = citation_precision(answer.citations, case.expected_paths)
        results.append(
            EvalCaseResult(
                question=case.question,
                retrieved_paths=paths,
                first_relevant_rank=rank,
                hit=rank is not None,
                substring_hit=substring_hit([h.chunk.text for h in hits], case.expected_substrings),
                citation_precision=precision,
            )
        )
        if progress:
            progress(n, len(cases))

    substring_results = [r.substring_hit for r in results if r.substring_hit is not None]
    precisions = [r.citation_precision for r in results if r.citation_precision is not None]
    return EvalReport(
        k=k,
        cases=results,
        hit_rate=mean([1.0 if r.hit else 0.0 for r in results]),
        mrr=mean(
            [
                reciprocal_rank(r.retrieved_paths, c.expected_paths, k)
                for r, c in zip(results, cases, strict=True)
            ]
        ),
        substring_hit_rate=mean([1.0 if s else 0.0 for s in substring_results])
        if substring_results
        else None,
        citation_precision=mean(precisions) if precisions else None,
    )
