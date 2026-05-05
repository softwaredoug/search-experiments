from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

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
        print(
            f"{record.get('round', '')}\t{mean_str}\t{delta_str}\t{short_name}\t{summary}"
        )
        if isinstance(mean_ndcg, (int, float)):
            prev = mean_ndcg


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_") or "run"


def _plot_rounds(
    records: list[dict],
    *,
    title: str,
    metric_key: str,
    output_path: Path,
    baseline: float | None = None,
    until_round: int | None = None,
) -> None:
    if not records:
        return
    if until_round is not None:
        records = [record for record in records if record.get("round") is not None and record.get("round") <= until_round]
        if not records:
            return
    rounds = [record.get("round") for record in records]
    ndcgs = [record.get(metric_key) for record in records]
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
    ax = plt.gca()
    plt.plot(rounds, ndcgs, color="#4c78a8", linewidth=2)
    plt.scatter(rounds, ndcgs, color=colors, s=60, zorder=3)

    y_limits = ax.get_ylim()
    if baseline is not None:
        ax.axhspan(0, baseline, color="#f1e4cf", alpha=0.35, zorder=0)
        ax.axhline(baseline, color="#a08b6c", linestyle=":", linewidth=1.5)
        ax.set_ylim(y_limits)

    annotations = []
    for record, x, y in zip(records, rounds, ndcgs):
        label = record.get("short_name") or f"round {record.get('round', '')}"
        if label == "baseline" and record.get("round") == 0:
            label = "start"
        if isinstance(y, (int, float)):
            annotations.append(
                plt.annotate(
                    label,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 0),
                    ha="right",
                    va="top",
                    rotation=-45,
                    rotation_mode="anchor",
                )
            )

    if annotations:
        fig = plt.gcf()
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for annotation in annotations:
            bbox = annotation.get_window_extent(renderer=renderer)
            width_points = bbox.width * 72.0 / fig.dpi
            height_points = bbox.height * 72.0 / fig.dpi
            annotation.set_position((width_points, -height_points))

    if baseline is not None:
        y_min, y_max = y_limits
        y_offset = (y_max - y_min) * 0.035
        label_y = max(y_min, baseline - y_offset)
        x_min, x_max = ax.get_xlim()
        ax.text(
            x_max,
            label_y,
            "BM25 baseline",
            ha="right",
            va="top",
            color="#a08b6c",
            alpha=0.6,
        )

    plt.title(title)
    plt.xlabel("Round")
    ylabel = "Mean NDCG" if metric_key == "mean_ndcg" else "Mean NDCG (test)"
    plt.ylabel(ylabel)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            print("Canceled.")
            return
    plt.savefig(output_path, dpi=150)
    plt.close()


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
    parser.add_argument(
        "--metric",
        choices=["full", "test"],
        default="full",
        help="Which NDCG series to plot (full or test).",
    )
    parser.add_argument(
        "--output-dir",
        default="assets",
        help="Directory to write plot image.",
    )
    parser.add_argument(
        "--title",
        help="Override chart title.",
    )
    parser.add_argument(
        "--until-round",
        type=int,
        help="Only plot rounds up to and including this value.",
    )
    args = parser.parse_args()

    run_path = (
        Path(args.run_path).expanduser()
        if args.run_path
        else _find_latest_run(args.dataset, args.strategy)
    )
    print(f"Run path: {run_path}")
    records = _load_rounds(run_path)
    _print_rounds(records)
    metric_key = "mean_ndcg" if args.metric == "full" else "mean_ndcg_test"
    baseline_map = {
        "wands": 0.5408,
        "esci": 0.2895,
    }
    baseline = None
    if metric_key == "mean_ndcg":
        baseline = baseline_map.get(args.dataset)
    title = args.title or f"Codegen run: {run_path} ({args.metric})"
    if args.title:
        filename = _slugify(args.title)
    else:
        strategy_name = args.strategy or run_path.parent.name
        filename = "_".join(
            [
                "codegen",
                _slugify(args.dataset),
                _slugify(strategy_name),
                _slugify(run_path.name),
                _slugify(args.metric),
            ]
        )
    output_path = Path(args.output_dir) / f"{filename}.png"
    _plot_rounds(
        records,
        title=title,
        metric_key=metric_key,
        output_path=output_path,
        baseline=baseline,
        until_round=args.until_round,
    )
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
