from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from exps.runners.run import RunParams, run_benchmark


def _collect_rounds(run_path: Path) -> list[int]:
    rounds = []
    for path in run_path.glob("reranker_round_*.py"):
        try:
            round_num = int(path.stem.split("_round_")[-1])
        except ValueError:
            continue
        rounds.append(round_num)
    return sorted(set(rounds))


def _append_rows(csv_path: Path, rows: list[dict[str, str | int | float | None]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay codegen rounds for msmarco/minimarco and append MRR stats."
    )
    parser.add_argument(
        "--strategy-path",
        default="configs/codegen/codegen_minimarco.yml",
        help="Strategy config to run.",
    )
    parser.add_argument(
        "--run-path",
        default="runs/codegen/minimarco/codegen_minimarco/20260504_033459",
        help="Codegen run directory containing reranker_round_*.py files.",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="CSV file to append results.",
    )
    parser.add_argument(
        "--datasets",
        default="msmarco,minimarco",
        help="Comma-separated list of datasets to run.",
    )
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_path = Path(args.run_path).expanduser()
    rounds = _collect_rounds(run_path)
    if not rounds:
        raise FileNotFoundError(f"No reranker_round_*.py files found in {run_path}")
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    output_path = Path(args.output).expanduser()

    rows: list[dict[str, str | int | float | None]] = []
    for round_num in rounds:
        for dataset in datasets:
            params = RunParams(
                strategy_path=args.strategy_path,
                base_path=None,
                dataset=dataset,
                codegen_run_round=round_num,
                num_queries=args.num_queries,
                seed=args.seed,
                workers=args.workers,
                device=args.device,
                no_cache=args.no_cache,
            )
            result = run_benchmark(params)
            metric_name = (result.metric_name or "metric").lower()
            summary = result.summary or {}
            rows.append(
                {
                    "timestamp": datetime.utcnow().strftime("%Y%m%d_%H%M%S"),
                    "dataset": dataset,
                    "round": round_num,
                    "metric": metric_name,
                    f"mean_{metric_name}": summary.get(f"mean_{metric_name}"),
                    f"median_{metric_name}": summary.get(f"median_{metric_name}"),
                    "strategy_path": args.strategy_path,
                    "run_path": str(run_path),
                }
            )
        _append_rows(output_path, rows)
        rows = []

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
