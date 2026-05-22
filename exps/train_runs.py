from __future__ import annotations

from pathlib import Path

from exps.run_dirs import make_strategy_run_dir


def make_train_run_dir(
    *,
    dataset: str,
    strategy_name: str,
    strategy_type: str,
    run_started_at: str | None = None,
) -> Path:
    return make_strategy_run_dir(
        dataset=dataset,
        strategy_name=strategy_name,
        strategy_type=strategy_type,
        run_started_at=run_started_at,
    )
