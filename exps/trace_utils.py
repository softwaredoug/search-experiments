from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from exps.paths import AGENTIC_TRACE_ROOT

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str, *, fallback: str = "query") -> str:
    slug = _SLUG_RE.sub("_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or fallback


def build_agentic_trace_root(
    strategy_name: str,
    dataset: str,
    *,
    run_started_at: str | None = None,
) -> Path:
    if run_started_at is None:
        run_started_at = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return (
        AGENTIC_TRACE_ROOT
        / slugify(dataset, fallback="dataset")
        / slugify(strategy_name, fallback="strategy")
        / run_started_at
    )


def dataset_from_trace_path(trace_path: Path) -> str | None:
    try:
        return trace_path.parent.parent.name
    except (AttributeError, IndexError):
        return None
