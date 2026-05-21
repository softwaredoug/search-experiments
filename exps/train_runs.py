from __future__ import annotations

from datetime import datetime
from pathlib import Path

from exps.paths import SEARCH_EXPERIMENTS_ROOT


def make_train_run_dir(
    *,
    dataset: str,
    strategy_name: str,
    strategy_type: str,
    run_started_at: str | None = None,
) -> Path:
    timestamp = run_started_at or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = SEARCH_EXPERIMENTS_ROOT / strategy_type / dataset / strategy_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
