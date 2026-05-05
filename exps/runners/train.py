from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from exps.codegen.train import train_codegen_strategy
from exps.datasets import DatasetName, get_dataset
from exps.strategy_config import load_strategy_config
from exps.strategy_factory import requires_bm25, strategy_params_for_config


class TrainParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_path: str
    base_path: str | None = None
    dataset: DatasetName = "wands"
    num_queries: int | None = None
    seed: int = 42
    workers: int = 1
    device: str | None = None
    rounds: int | None = None


class TrainResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_name: str
    strategy_params: dict
    artifact_path: str
    metadata: dict


def _write_summary_csv(path: str, *, dataset: str, result: TrainResult) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_sha = ""
    row = {
        "dataset": dataset,
        "commit_sha": commit_sha,
        "strategy_name": result.strategy_name,
        "strategy_params": json.dumps(result.strategy_params, sort_keys=True),
        "codegen_artifact_path": result.artifact_path,
        "train_metadata": json.dumps(result.metadata, sort_keys=True),
    }

    file_exists = output_path.exists()
    fieldnames = None
    if file_exists:
        with output_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames

    if not fieldnames:
        fieldnames = list(row.keys())

    write_header = not file_exists or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def train_strategy(params: TrainParams) -> TrainResult:
    strategy_config = load_strategy_config(params.strategy_path, base_path=params.base_path)
    if strategy_config.type != "codegen":
        raise ValueError("Training is only supported for codegen strategies.")
    strategy_params = strategy_params_for_config(strategy_config, device=params.device)
    train_params = dict(strategy_params.get("train") or {})
    if params.rounds is not None:
        train_params["rounds"] = params.rounds
    strategy_params["train"] = train_params

    dataset = get_dataset(
        params.dataset,
        workers=params.workers,
        ensure_snowball=requires_bm25(strategy_config.type, strategy_params),
    )
    artifact = train_codegen_strategy(
        strategy_name=strategy_config.name,
        dataset=params.dataset,
        corpus=dataset.corpus,
        judgments=dataset.judgments,
        params=strategy_params,
        run_path=strategy_config.path,
        device=params.device,
        workers=params.workers,
        report_num_queries=params.num_queries,
        report_seed=params.seed,
    )
    metadata_path = artifact.path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return TrainResult(
        strategy_name=strategy_config.name,
        strategy_params=strategy_params,
        artifact_path=str(artifact.path),
        metadata=metadata,
    )
