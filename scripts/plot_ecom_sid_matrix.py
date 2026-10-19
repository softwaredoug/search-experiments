#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASETS = ("wands", "esci")
STRATEGIES = (
    ("bm25", "BM25"),
    ("agentic_bm25_ecommerce", "GPT-5 · BM25"),
    ("agentic_bm25_e5_ecommerce_gpt5", "GPT-5 · BM25 + E5"),
    ("agentic_bm25_ecommerce_gpt5_mini", "GPT-5-mini · BM25"),
    ("agentic_bm25_e5_ecommerce_gpt5_mini", "GPT-5-mini · BM25 + E5"),
    ("agentic_ecom_sid_bm25_low", "SID-2 · BM25 · low"),
    ("agentic_ecom_sid_bm25_medium", "SID-2 · BM25 · medium"),
    ("agentic_ecom_sid_bm25_high", "SID-2 · BM25 · high"),
    ("agentic_ecom_sid_bm25_e5_base_v2_low", "SID-2 · BM25 + E5 · low"),
    (
        "agentic_ecom_sid_bm25_e5_base_v2_medium",
        "SID-2 · BM25 + E5 · medium",
    ),
    ("agentic_ecom_sid_bm25_e5_base_v2_high", "SID-2 · BM25 + E5 · high"),
)


def _read_mean_ndcg(path: Path) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    strategy_names = {name for name, _ in STRATEGIES}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "strategy_name", "metric_name", "mean_ndcg"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))} columns.")
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            if (
                dataset not in DATASETS
                or strategy not in strategy_names
                or row["metric_name"].lower() != "ndcg"
            ):
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


def _read_mean_tool_calls(path: Path) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    strategy_names = {name for name, _ in STRATEGIES}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "strategy_name", "metric_name", "mean_resource.tool_calls"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))} columns.")
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            calls = row["mean_resource.tool_calls"]
            if (
                dataset not in DATASETS
                or strategy not in strategy_names
                or row["metric_name"].lower() != "ndcg"
                or not calls
            ):
                continue
            try:
                totals[(dataset, strategy)].append(float(calls))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mean_resource.tool_calls for dataset={dataset!r}, "
                    f"strategy={strategy!r}."
                ) from exc

    return {
        key: sum(values) / len(values)
        for key, values in totals.items()
        if values
    }


def _pareto_front(
    points: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    """Return points not dominated on retrieval (higher) and tool calls (lower)."""
    frontier = []
    for point in points:
        ndcg = float(point["mean_ndcg"])
        calls = float(point["mean_resource.tool_calls"])
        dominated = any(
            float(other["mean_ndcg"]) >= ndcg
            and float(other["mean_resource.tool_calls"]) <= calls
            and (
                float(other["mean_ndcg"]) > ndcg
                or float(other["mean_resource.tool_calls"]) < calls
            )
            for other in points
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: float(point["mean_ndcg"]))


def _avoid_label_overlaps(annotations, ax, fig) -> None:
    """Move Pareto labels vertically until each is readable and inside its axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bounds = ax.get_window_extent(renderer)
    placed = []
    offsets = [0, *(offset for step in range(1, 10) for offset in (step * 10, -step * 10))]
    for annotation in annotations:
        x, original_y = annotation.get_position()
        for offset in offsets:
            annotation.set_position((x, original_y + offset))
            bounds = annotation.get_window_extent(renderer)
            inside_axes = bounds.y0 >= axes_bounds.y0 and bounds.y1 <= axes_bounds.y1
            if inside_axes and not any(bounds.overlaps(previous) for previous in placed):
                placed.append(bounds)
                break


def plot_results(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path)
    labels = dict(STRATEGIES)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(24, 7), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        ordered = sorted(
            ((strategy, means[(dataset, strategy)]) for strategy, _ in STRATEGIES),
            key=lambda item: item[1],
        )
        positions = list(range(len(ordered)))
        values = [value for _, value in ordered]
        ax.plot(positions, values, color="#4C78A8", marker="o", linewidth=2)
        for position, value in zip(positions, values):
            ax.annotate(
                f"{value:.3f}",
                (position, value),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(dataset.upper())
        ax.set_xticks(positions)
        ax.set_xticklabels(
            [labels[strategy] for strategy, _ in ordered], rotation=45, ha="right"
        )
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Mean NDCG")
    fig.suptitle("E-commerce retrieval quality by model, tools, and SID-2 reasoning effort")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pareto_results(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path)
    tool_calls = _read_mean_tool_calls(input_path)
    labels = dict(STRATEGIES)
    pareto_labels = {
        strategy: (
            label.replace("GPT-5-mini", "GPT-5m")
            .replace("SID-2 · ", "SID ")
            .replace(" · ", " ")
            .replace("BM25 + E5", "BM25+E5")
        )
        for strategy, label in STRATEGIES
    }
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(16, 6), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        points = [
            {
                "strategy": strategy,
                "label": labels[strategy],
                "mean_ndcg": means[(dataset, strategy)],
                "mean_resource.tool_calls": tool_calls[(dataset, strategy)],
            }
            for strategy, _ in STRATEGIES
            if (dataset, strategy) in tool_calls
        ]
        frontier = _pareto_front(points)
        frontier_names = {str(point["strategy"]) for point in frontier}
        annotations = []
        for point in points:
            ndcg = float(point["mean_ndcg"])
            calls = float(point["mean_resource.tool_calls"])
            is_frontier = str(point["strategy"]) in frontier_names
            ax.scatter(
                ndcg,
                calls,
                color="#F58518" if is_frontier else "#9E9E9E",
                s=55,
                zorder=3 if is_frontier else 2,
            )
            annotations.append(
                ax.annotate(
                    pareto_labels[str(point["strategy"])],
                    (ndcg, calls),
                    textcoords="offset points",
                    xytext=(5, 5),
                    ha="left",
                    fontsize=8,
                )
            )
        if frontier:
            ax.plot(
                [float(point["mean_ndcg"]) for point in frontier],
                [float(point["mean_resource.tool_calls"]) for point in frontier],
                color="#F58518",
                linewidth=2,
                marker="o",
                zorder=4,
            )
        _avoid_label_overlaps(annotations, ax, fig)
        ax.set_title(dataset.upper())
        ax.set_xlabel("Mean NDCG (higher is better)")
        ax.grid(linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Mean Tool Calls per Query (lower is better)")
    fig.suptitle("E-commerce tool-call Pareto frontier")
    fig.text(0.5, 0.01, "BM25 omitted: tool-call data unavailable", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot e-commerce SID matrix retrieval quality and tool-call Pareto."
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
