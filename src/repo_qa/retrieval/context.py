"""Build the prompt context: merge overlapping hits per file, then fill a token budget."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from repo_qa.models import ContextItem, ContextPack, SearchHit
from repo_qa.text import estimate_tokens


def item_header(item: ContextItem) -> str:
    symbols = f" ({', '.join(item.symbols)})" if item.symbols else ""
    return f"[{item.location}]{symbols}"


def item_tokens(item: ContextItem) -> int:
    return estimate_tokens(item_header(item)) + estimate_tokens(item.text) + 2


def merge_hits(hits: Sequence[SearchHit]) -> list[ContextItem]:
    """Merge hits whose line ranges overlap or touch within the same file.

    Chunks are exact line slices of their file, so merged text is stitched line by line.
    A merged region keeps the best score among its parts.
    """
    by_path: dict[str, list[SearchHit]] = defaultdict(list)
    for hit in hits:
        by_path[hit.chunk.path].append(hit)

    items: list[ContextItem] = []
    for path, group in by_path.items():
        group.sort(key=lambda h: (h.chunk.start_line, h.chunk.end_line))
        current: list[SearchHit] = []
        cur_end = -1
        for hit in group:
            if current and hit.chunk.start_line <= cur_end + 1:
                current.append(hit)
                cur_end = max(cur_end, hit.chunk.end_line)
            else:
                if current:
                    items.append(_merge(path, current))
                current, cur_end = [hit], hit.chunk.end_line
        if current:
            items.append(_merge(path, current))
    return items


def _merge(path: str, hits: list[SearchHit]) -> ContextItem:
    lines: dict[int, str] = {}
    for hit in hits:
        for offset, line in enumerate(hit.chunk.text.split("\n")):
            lines.setdefault(hit.chunk.start_line + offset, line)
    start, end = min(lines), max(lines)
    symbols = list(dict.fromkeys(h.chunk.symbol for h in hits if h.chunk.symbol))
    return ContextItem(
        path=path,
        language=hits[0].chunk.language,
        start_line=start,
        end_line=end,
        text="\n".join(lines.get(i, "") for i in range(start, end + 1)),
        score=max(h.score for h in hits),
        symbols=symbols,
        chunk_ids=list(dict.fromkeys(h.chunk.id for h in hits)),
    )


def _truncate(item: ContextItem, budget: int) -> ContextItem | None:
    """Keep as many leading lines of ``item`` as fit in ``budget`` tokens."""
    lines = item.text.split("\n")
    lo, hi = 0, len(lines)
    while lo < hi:  # largest prefix that fits
        mid = (lo + hi + 1) // 2
        candidate = item.model_copy(
            update={"text": "\n".join(lines[:mid]), "end_line": item.start_line + mid - 1}
        )
        if item_tokens(candidate) <= budget:
            lo = mid
        else:
            hi = mid - 1
    if lo == 0:
        return None
    return item.model_copy(
        update={"text": "\n".join(lines[:lo]), "end_line": item.start_line + lo - 1}
    )


def build_context_pack(hits: Sequence[SearchHit], token_budget: int) -> ContextPack:
    """Greedy best-first packing; items stay in descending score order.

    If even the best item does not fit, its leading lines are kept so the pack is never
    empty while there are hits.
    """
    candidates = sorted(merge_hits(hits), key=lambda i: (-i.score, i.path, i.start_line))
    chosen: list[ContextItem] = []
    used = 0
    dropped = 0
    for item in candidates:
        cost = item_tokens(item)
        if used + cost <= token_budget:
            chosen.append(item)
            used += cost
        elif not chosen and (cut := _truncate(item, token_budget)) is not None:
            chosen.append(cut)
            used += item_tokens(cut)
        else:
            dropped += 1
    return ContextPack(items=chosen, token_count=used, token_budget=token_budget, dropped=dropped)
