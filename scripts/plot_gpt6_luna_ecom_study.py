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


DATASETS = ("wands", "esci")
STRATEGIES = (
    ("bm25", "BM25"),
    ("agentic_gpt5_mini_bm25", "GPT-5-mini · BM25"),
    ("agentic_gpt5_mini_bm25_e5", "GPT-5-mini · BM25 + E5"),
    ("agentic_gpt6_luna_bm25", "GPT-6 Luna · BM25"),
    ("agentic_gpt6_luna_bm25_e5", "GPT-6 Luna · BM25 + E5"),
)
MODEL_BY_STRATEGY = {
    "agentic_gpt5_mini_bm25": "gpt-5-mini",
    "agentic_gpt5_mini_bm25_e5": "gpt-5-mini",
    "agentic_gpt6_luna_bm25": "gpt-6-luna",
    "agentic_gpt6_luna_bm25_e5": "gpt-6-luna",
}
TOKEN_PRICES_PER_MILLION = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    "gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "output": 0.50},
}
MODEL_COLORS = {"gpt-5-mini": "#4C78A8", "gpt-6-luna": "#F58518"}


def _read_mean_ndcg(path: Path) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    strategy_names = {strategy for strategy, _ in STRATEGIES}
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
                or not row["mean_ndcg"]
            ):
                continue
            try:
                totals[(dataset, strategy)].append(float(row["mean_ndcg"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mean_ndcg for dataset={dataset!r}, strategy={strategy!r}."
                ) from exc

    return {key: sum(values) / len(values) for key, values in totals.items()}


def _read_mean_token_cost_usd(path: Path) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "strategy_name",
            "metric_name",
            "mean_resource.input_tokens",
            "mean_resource.output_tokens",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))} columns.")
        has_cached_tokens = "mean_resource.cached_input_tokens" in reader.fieldnames
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            model = MODEL_BY_STRATEGY.get(strategy)
            input_tokens = row["mean_resource.input_tokens"]
            output_tokens = row["mean_resource.output_tokens"]
            if (
                dataset not in DATASETS
                or row["metric_name"].lower() != "ndcg"
                or model is None
                or not input_tokens
                or not output_tokens
            ):
                continue
            try:
                input_count = float(input_tokens)
                output_count = float(output_tokens)
                cached_count = (
                    float(row["mean_resource.cached_input_tokens"] or 0)
                    if has_cached_tokens
                    else 0.0
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid token usage for dataset={dataset!r}, strategy={strategy!r}."
                ) from exc
            if input_count < 0 or output_count < 0 or cached_count < 0:
                raise ValueError(
                    f"Negative token usage for dataset={dataset!r}, strategy={strategy!r}."
                )
            cached_count = min(cached_count, input_count)
            prices = TOKEN_PRICES_PER_MILLION[model]
            cost_usd = (
                (input_count - cached_count) * prices["input"]
                + cached_count * prices["cached_input"]
                + output_count * prices["output"]
            ) / 1_000_000
            totals[(dataset, strategy)].append(cost_usd)

    return {key: sum(values) / len(values) for key, values in totals.items()}


def _pareto_front(points: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    """Return points not dominated on retrieval (higher) and token cost (lower)."""
    frontier = []
    for point in points:
        ndcg = float(point["mean_ndcg"])
        cost = float(point["mean_token_cost_usd"])
        dominated = any(
            float(other["mean_ndcg"]) >= ndcg
            and float(other["mean_token_cost_usd"]) <= cost
            and (
                float(other["mean_ndcg"]) > ndcg
                or float(other["mean_token_cost_usd"]) < cost
            )
            for other in points
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: float(point["mean_token_cost_usd"]))


def plot_retrieval(input_path: Path, output_path: Path, dataset: str) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path)
    labels = dict(STRATEGIES)
    ordered = sorted(
        (
            (strategy, means[(dataset, strategy)])
            for strategy, _ in STRATEGIES
            if (dataset, strategy) in means
        ),
        key=lambda item: item[1],
    )
    if not ordered:
        raise ValueError(f"No mean NDCG results found for dataset={dataset!r}.")

    positions = list(range(len(ordered)))
    values = [value for _, value in ordered]
    fig, ax = plt.subplots(figsize=(11, 6))
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
        [labels[strategy] for strategy, _ in ordered], rotation=35, ha="right"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean NDCG@10")
    ceiling = read_mean_max_ndcg(input_path)
    if dataset in ceiling:
        add_mean_max_ndcg_line(ax, ceiling[dataset])
        ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.suptitle("E-commerce retrieval quality: BM25, GPT-5-mini, and GPT-6 Luna")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_cost_pareto(input_path: Path, output_path: Path, dataset: str) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path)
    costs = _read_mean_token_cost_usd(input_path)
    labels = dict(STRATEGIES)
    strategies = [strategy for strategy, _ in STRATEGIES if strategy in MODEL_BY_STRATEGY]
    points = [
        {
            "strategy": strategy,
            "mean_ndcg": means[(dataset, strategy)],
            "mean_token_cost_usd": costs[(dataset, strategy)],
        }
        for strategy in strategies
        if (dataset, strategy) in means and (dataset, strategy) in costs
    ]
    if not points:
        raise ValueError(f"No token-cost and NDCG results found for dataset={dataset!r}.")

    frontier = _pareto_front(points)
    frontier_names = {str(point["strategy"]) for point in frontier}
    fig, ax = plt.subplots(figsize=(10, 6))
    annotations = []
    for point in points:
        strategy = str(point["strategy"])
        model = MODEL_BY_STRATEGY[strategy]
        label = labels[strategy]
        marker = "s" if "_e5" in strategy else "o"
        is_frontier = strategy in frontier_names
        cost = float(point["mean_token_cost_usd"])
        ndcg = float(point["mean_ndcg"])
        ax.scatter(
            cost,
            ndcg,
            color=MODEL_COLORS[model],
            marker=marker,
            s=65,
            edgecolors="#222222" if is_frontier else "none",
            linewidths=0.8,
            zorder=3 if is_frontier else 2,
        )
        annotations.append(
            ax.annotate(
                label.replace("GPT-5-mini", "GPT-5m").replace(" · ", " "),
                (cost, ndcg),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=8,
            )
        )
    if frontier:
        ax.plot(
            [float(point["mean_token_cost_usd"]) for point in frontier],
            [float(point["mean_ndcg"]) for point in frontier],
            color="#F58518",
            linewidth=2,
            marker="o",
            zorder=4,
        )

    ceiling = read_mean_max_ndcg(input_path)
    if dataset in ceiling:
        add_mean_max_ndcg_line(ax, ceiling[dataset])
    ax.set_title(dataset.upper())
    ax.set_xlabel("Estimated model token cost per query (USD; lower is better)")
    ax.set_ylabel("Mean NDCG@10 (higher is better)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.suptitle("GPT-5-mini and GPT-6 Luna e-commerce cost Pareto")
    rates = (
        "GPT-6 Luna: $0.10/M input, $0.01/M cached input, $0.50/M output; "
        "GPT-5-mini: $0.25/M input, $0.025/M cached input, $2.00/M output"
    )
    fig.text(0.5, 0.01, rates, ha="center", fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot e-commerce retrieval quality and GPT-5-mini/GPT-6 Luna token-cost "
            "Pareto frontiers for WANDS and ESCI."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.output_prefix.with_suffix("") if args.output_prefix.suffix else args.output_prefix
    for dataset in DATASETS:
        retrieval_output = prefix.parent / f"{prefix.name}_{dataset}.png"
        cost_output = prefix.parent / f"{prefix.name}_cost_pareto_{dataset}.png"
        plot_retrieval(args.input, retrieval_output, dataset)
        plot_cost_pareto(args.input, cost_output, dataset)
        print(f"Wrote {retrieval_output}")
        print(f"Wrote {cost_output}")


if __name__ == "__main__":
    main()
