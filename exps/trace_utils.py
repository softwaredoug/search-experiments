from __future__ import annotations

from pathlib import Path

from exps.paths import make_strategy_run_dir


def build_agentic_trace_root(
    strategy_name: str,
    dataset: str,
    *,
    run_started_at: str | None = None,
) -> Path:
    return make_strategy_run_dir(
        dataset=dataset,
        strategy_name=strategy_name,
        strategy_type="agentic",
        run_started_at=run_started_at,
    )
