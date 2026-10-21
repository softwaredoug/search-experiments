#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    from ._plot_helpers import add_mean_max_ndcg_line, read_mean_max_ndcg
except ImportError:  # Script execution from the scripts directory.
    from _plot_helpers import add_mean_max_ndcg_line, read_mean_max_ndcg

DATASETS = ("esci", "wands")
STRATEGIES = (
    ("bm25", "BM25"),
    ("embedding_e5", "E5-base-v2"),
    ("agentic_bm25_e5_ecommerce_gpt5_mini", "Agentic BM25 + E5"),
    ("agentic_ecom_bm25_fewshot_judge_jev", "Few-shot + Jev judge"),
    ("agentic_ecom_bm25_fewshot_judge", "Few-shot + GPT-5-mini judge"),
)


def _read_mean_ndcg(path: Path) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"dataset", "strategy_name", "mean_ndcg"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{path} must contain dataset, strategy_name, and mean_ndcg columns."
            )
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            if dataset not in DATASETS or strategy not in {name for name, _ in STRATEGIES}:
                continue
            try:
                totals[(dataset, strategy)].append(float(row["mean_ndcg"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mean_ndcg for dataset={dataset!r}, strategy={strategy!r}."
                ) from exc

    means = {
        key: sum(values) / len(values)
        for key, values in totals.items()
        if values
    }
    missing = [
        (dataset, strategy)
        for dataset in DATASETS
        for strategy, _ in STRATEGIES
        if (dataset, strategy) not in means
    ]
    if missing:
        missing_text = ", ".join(f"{dataset}/{strategy}" for dataset, strategy in missing)
        raise ValueError(f"No mean NDCG results found for: {missing_text}")
    return means


def _read_pareto_points(
    summary_path: Path,
) -> dict[str, list[dict[str, float | str]]]:
    ndcg_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    tool_call_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    with summary_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row.get("dataset", "")
            strategy = row.get("strategy_name", "")
            if dataset not in DATASETS or strategy not in {
                name for name, _ in STRATEGIES
            }:
                continue
            if row.get("metric_name", "").lower() != "ndcg":
                continue
            try:
                ndcg_totals[(dataset, strategy)].append(float(row["mean_ndcg"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mean_ndcg for {dataset}/{strategy} in {summary_path}"
                ) from exc
            calls = row.get("mean_resource.tool_calls")
            try:
                tool_call_totals[(dataset, strategy)].append(float(calls))
            except (TypeError, ValueError):
                pass

    labels = dict(STRATEGIES)
    points_by_dataset: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for (dataset, strategy), ndcg_values in ndcg_totals.items():
        calls = tool_call_totals.get((dataset, strategy), [])
        if not calls:
            continue
        mean_calls = sum(calls) / len(calls)
        if mean_calls <= 0:
            continue
        points_by_dataset[dataset].append(
            {
                "strategy": strategy,
                "label": labels[strategy],
                "mean_ndcg": sum(ndcg_values) / len(ndcg_values),
                "mean_resource.tool_calls": mean_calls,
            }
        )
    return points_by_dataset


def _pareto_front(
    points: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    ordered = sorted(
        points,
        key=lambda row: (
            float(row["mean_resource.tool_calls"]),
            -float(row["mean_ndcg"]),
        ),
    )
    frontier = []
    best_ndcg = float("-inf")
    for row in ordered:
        ndcg = float(row["mean_ndcg"])
        if ndcg > best_ndcg:
            frontier.append(row)
            best_ndcg = ndcg
    return frontier


def plot_results(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path)
    max_ndcg_by_dataset = read_mean_max_ndcg(input_path)
    labels = [label for _, label in STRATEGIES]
    positions = range(len(STRATEGIES))
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#54A24B", "#E45756"]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        values = [means[(dataset, strategy)] for strategy, _ in STRATEGIES]
        bars = ax.bar(positions, values, color=colors)
        for bar, value in zip(bars, values):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(dataset.upper())
        ax.set_xticks(list(positions))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        if dataset in max_ndcg_by_dataset:
            add_mean_max_ndcg_line(ax, max_ndcg_by_dataset[dataset])
            ax.legend(loc="best", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean NDCG")
    fig.suptitle("E-commerce search judge variants")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pareto_results(
    summary_path: Path,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    points_by_dataset = _read_pareto_points(summary_path)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        points = points_by_dataset.get(dataset, [])
        frontier = _pareto_front(points)
        frontier_names = {str(point["strategy"]) for point in frontier}
        for point in points:
            is_frontier = str(point["strategy"]) in frontier_names
            ax.scatter(
                point["mean_resource.tool_calls"],
                point["mean_ndcg"],
                color="#F58518" if is_frontier else "#9E9E9E",
                s=65,
                zorder=3 if is_frontier else 2,
            )
            ax.annotate(
                str(point["label"]),
                (point["mean_resource.tool_calls"], point["mean_ndcg"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        if frontier:
            ax.plot(
                [point["mean_resource.tool_calls"] for point in frontier],
                [point["mean_ndcg"] for point in frontier],
                color="#F58518",
                linewidth=2,
                marker="o",
                zorder=4,
            )
        ax.set_title(dataset.upper())
        ax.set_xlabel("Mean Tool Calls per Query")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean NDCG")
    fig.suptitle("E-commerce Judge Variants: Tool Calls vs NDCG Pareto")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot e-commerce judge variant NDCG and tool-call Pareto."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pareto-output", type=Path)
    args = parser.parse_args()
    plot_results(args.input, args.output)
    if args.pareto_output is not None:
        plot_pareto_results(args.input, args.pareto_output)
    print(f"Wrote {args.output}")
    if args.pareto_output is not None:
        print(f"Wrote {args.pareto_output}")


if __name__ == "__main__":
    main()
