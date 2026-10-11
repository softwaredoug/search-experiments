from __future__ import annotations

from pathlib import Path

from exps.paths import SEARCH_EXPERIMENTS_ROOT, make_strategy_run_dir, slugify


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


def find_latest_train_run(
    *, dataset: str, strategy_name: str, strategy_type: str
) -> Path | None:
    runs_root = (
        SEARCH_EXPERIMENTS_ROOT
        / slugify(strategy_type, fallback="strategy")
        / slugify(dataset, fallback="dataset")
        / slugify(strategy_name, fallback="strategy")
    )
    if not runs_root.exists():
        return None
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]
