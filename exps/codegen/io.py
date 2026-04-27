from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from exps.paths import SEARCH_EXPERIMENTS_ROOT


CODEGEN_ROOT = SEARCH_EXPERIMENTS_ROOT / "codegen"


def make_codegen_dir(
    dataset: str, strategy_name: str, *, run_started_at: str | None = None
) -> Path:
    timestamp = run_started_at or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = CODEGEN_ROOT / dataset / strategy_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def reranker_path(output_dir: Path) -> Path:
    return output_dir / "reranker.py"


def metadata_path(output_dir: Path) -> Path:
    return output_dir / "metadata.json"


def write_metadata(output_dir: Path, payload: dict) -> None:
    path = metadata_path(output_dir)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
