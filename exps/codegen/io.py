from __future__ import annotations

import json
from pathlib import Path

from exps.paths import SEARCH_EXPERIMENTS_ROOT
from exps.paths import make_strategy_run_dir


CODEGEN_ROOT = SEARCH_EXPERIMENTS_ROOT / "codegen"


def find_latest_codegen_run(dataset: str, strategy_name: str) -> Path | None:
    runs_root = CODEGEN_ROOT / dataset / strategy_name
    if not runs_root.exists():
        return None
    candidates = [path for path in runs_root.iterdir() if path.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[-1]


def make_codegen_dir(
    dataset: str, strategy_name: str, *, run_started_at: str | None = None
) -> Path:
    return make_strategy_run_dir(
        dataset=dataset,
        strategy_name=strategy_name,
        strategy_type="codegen",
        run_started_at=run_started_at,
    )


def reranker_path(output_dir: Path) -> Path:
    return output_dir / "reranker.py"


def metadata_path(output_dir: Path) -> Path:
    return output_dir / "metadata.json"


def write_metadata(output_dir: Path, payload: dict) -> None:
    path = metadata_path(output_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
