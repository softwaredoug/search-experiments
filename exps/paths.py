from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

SEARCH_EXPERIMENTS_ROOT = Path.home() / ".search-experiments"
AGENTIC_TRACE_ROOT = SEARCH_EXPERIMENTS_ROOT / "agentic" / "traces"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, fallback: str = "query") -> str:
    slug = _SLUG_RE.sub("_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or fallback


def dataset_from_trace_path(trace_path: Path) -> str | None:
    try:
        return trace_path.parent.parent.name
    except (AttributeError, IndexError):
        return None


def make_strategy_run_dir(
    *,
    dataset: str,
    strategy_name: str,
    strategy_type: str,
    run_started_at: str | None = None,
) -> Path:
    timestamp = run_started_at or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        SEARCH_EXPERIMENTS_ROOT
        / slugify(strategy_type, fallback="strategy")
        / slugify(dataset, fallback="dataset")
        / slugify(strategy_name, fallback="strategy")
        / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
