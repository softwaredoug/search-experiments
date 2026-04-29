from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


CODEGEN_ROOT = Path.home() / ".search-experiments" / "codegen"


def _find_latest_run(dataset: str, strategy: str | None) -> Path:
    dataset_root = CODEGEN_ROOT / dataset
    if not dataset_root.exists():
        raise FileNotFoundError(f"No codegen runs found for dataset: {dataset}")

    strategy_dirs = [dataset_root / strategy] if strategy else dataset_root.iterdir()
    latest_dir = None
    latest_mtime = None
    for strategy_dir in strategy_dirs:
        if not strategy_dir.exists() or not strategy_dir.is_dir():
            if strategy:
                raise FileNotFoundError(
                    f"No codegen runs found for dataset {dataset} and strategy {strategy}"
                )
            continue
        for run_dir in strategy_dir.iterdir():
            if not run_dir.is_dir():
                continue
            mtime = run_dir.stat().st_mtime
            if latest_mtime is None or mtime > latest_mtime:
                latest_mtime = mtime
                latest_dir = run_dir

    if latest_dir is None:
        raise FileNotFoundError(f"No codegen runs found for dataset: {dataset}")
    return latest_dir


def _load_rounds(run_path: Path) -> list[dict]:
    rounds_path = run_path / "rounds.jsonl"
    if rounds_path.exists():
        records = []
        for line in rounds_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
        return records

    metadata_path = run_path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summaries = metadata.get("round_summaries") or []
        if summaries:
            return summaries
        ndcgs = metadata.get("round_ndcgs") or []
        return [
            {"round": idx + 1, "mean_ndcg": value}
            for idx, value in enumerate(ndcgs)
        ]

    raise FileNotFoundError("No rounds.jsonl or metadata.json found in run path.")


def _print_rounds(records: list[dict]) -> None:
    if not records:
        print("No rounds to report.")
        return
    print("Round\tMeanNDCG\tDelta\tShortName\tSummary")
    prev = None
    for record in records:
        mean_ndcg = record.get("mean_ndcg")
        delta = None
        if isinstance(mean_ndcg, (int, float)) and isinstance(prev, (int, float)):
            delta = mean_ndcg - prev
        delta_str = f"{delta:+.4f}" if delta is not None else ""
        short_name = record.get("short_name") or ""
        summary = (record.get("summary") or "").replace("\n", " ")
        mean_str = f"{mean_ndcg:.4f}" if isinstance(mean_ndcg, (int, float)) else ""
        print(f"{record.get('round','')}\t{mean_str}\t{delta_str}\t{short_name}\t{summary}")
        if isinstance(mean_ndcg, (int, float)):
            prev = mean_ndcg


def _plot_rounds(records: list[dict], *, title: str) -> None:
    if not records:
        return
    rounds = [record.get("round") for record in records]
    ndcgs = [record.get("mean_ndcg") for record in records]
    deltas = [
        None if idx == 0 or ndcgs[idx] is None or ndcgs[idx - 1] is None else ndcgs[idx] - ndcgs[idx - 1]
        for idx in range(len(ndcgs))
    ]

    colors = []
    for delta in deltas:
        if delta is None:
            colors.append("#4c78a8")
        elif delta >= 0:
            colors.append("#2ca02c")
        else:
            colors.append("#d62728")

    plt.figure(figsize=(10, 6))
    plt.plot(rounds, ndcgs, color="#4c78a8", linewidth=2)
    plt.scatter(rounds, ndcgs, color=colors, s=60, zorder=3)

    for record, x, y in zip(records, rounds, ndcgs):
        label = record.get("short_name") or f"round {record.get('round', '')}"
        if isinstance(y, (int, float)):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 8), ha="center")

    plt.title(title)
    plt.xlabel("Round")
    plt.ylabel("Mean NDCG")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a codegen training run.")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., wands).")
    parser.add_argument(
        "--strategy",
        help="Strategy name to select latest run (e.g., codegen_sample).",
    )
    parser.add_argument(
        "--run-path",
        help="Path to the codegen run directory. Defaults to latest run.",
    )
    args = parser.parse_args()

    run_path = (
        Path(args.run_path).expanduser()
        if args.run_path
        else _find_latest_run(args.dataset, args.strategy)
    )
    records = _load_rounds(run_path)
    _print_rounds(records)
    title = f"Codegen run: {run_path}"
    _plot_rounds(records, title=title)


if __name__ == "__main__":
    main()
