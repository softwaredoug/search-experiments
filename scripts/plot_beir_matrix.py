#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASETS = ("scifact", "trec_covid")
STRATEGIES = (
    ("bm25", "BM25"),
    ("agentic_bm25_ecommerce_gpt5_mini", "GPT-5-mini · BM25"),
    ("agentic_bm25_e5_ecommerce_gpt5_mini", "GPT-5-mini · BM25 + E5"),
    ("agentic_bm25_ecommerce_gpt6_luna", "GPT-6 Luna · BM25"),
    ("agentic_bm25_e5_ecommerce_gpt6_luna", "GPT-6 Luna · BM25 + E5"),
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
MODEL_PRICES = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    "gpt-6-luna": {"input": 0.1, "cached_input": 0.01, "output": 0.5},
    "sid-2": {"input": 0.25, "cached_input": 0.025, "output": 1.0},
}
STRATEGY_MODELS = {
    "agentic_bm25_ecommerce_gpt5_mini": "gpt-5-mini",
    "agentic_bm25_e5_ecommerce_gpt5_mini": "gpt-5-mini",
    "agentic_bm25_ecommerce_gpt6_luna": "gpt-6-luna",
    "agentic_bm25_e5_ecommerce_gpt6_luna": "gpt-6-luna",
    "agentic_ecom_sid_bm25_low": "sid-2",
    "agentic_ecom_sid_bm25_medium": "sid-2",
    "agentic_ecom_sid_bm25_high": "sid-2",
    "agentic_ecom_sid_bm25_e5_base_v2_low": "sid-2",
    "agentic_ecom_sid_bm25_e5_base_v2_medium": "sid-2",
    "agentic_ecom_sid_bm25_e5_base_v2_high": "sid-2",
}


def _read_metric(
    path: Path,
    column: str,
    *,
    model_cost: bool = False,
) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    strategy_names = {name for name, _ in STRATEGIES}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "strategy_name", "metric_name", column}
        if model_cost:
            required.update({"mean_resource.input_tokens", "mean_resource.output_tokens"})
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))} columns.")
        has_cached_tokens = "mean_resource.cached_input_tokens" in reader.fieldnames
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            if (
                dataset not in DATASETS
                or strategy not in strategy_names
                or row["metric_name"].lower() != "ndcg"
            ):
                continue
            if model_cost:
                model = STRATEGY_MODELS.get(strategy)
                if model is None:
                    continue
                input_tokens = row["mean_resource.input_tokens"]
                output_tokens = row["mean_resource.output_tokens"]
                if not input_tokens or not output_tokens:
                    continue
                cached_tokens = (
                    row.get("mean_resource.cached_input_tokens", "")
                    if has_cached_tokens
                    else "0"
                )
                try:
                    input_count = float(input_tokens)
                    output_count = float(output_tokens)
                    cached_count = float(cached_tokens or 0)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid token usage for dataset={dataset!r}, "
                        f"strategy={strategy!r}."
                    ) from exc
                cached_count = min(max(cached_count, 0.0), input_count)
                prices = MODEL_PRICES[model]
                value = (
                    (input_count - cached_count) * prices["input"]
                    + cached_count * prices["cached_input"]
                    + output_count * prices["output"]
                ) / 1_000_000
            else:
                raw_value = row.get(column)
                if not raw_value:
                    continue
                try:
                    value = float(raw_value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid {column} for dataset={dataset!r}, "
                        f"strategy={strategy!r}."
                    ) from exc
            totals[(dataset, strategy)].append(value)

    return {
        key: sum(values) / len(values)
        for key, values in totals.items()
        if values
    }


def _pareto_front(
    points: list[dict[str, float | str]], resource_key: str
) -> list[dict[str, float | str]]:
    frontier = []
    for point in points:
        value = float(point[resource_key])
        ndcg = float(point["mean_ndcg"])
        dominated = any(
            float(other["mean_ndcg"]) >= ndcg
            and float(other[resource_key]) <= value
            and (
                float(other["mean_ndcg"]) > ndcg
                or float(other[resource_key]) < value
            )
            for other in points
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return sorted(frontier, key=lambda point: float(point[resource_key]))


def plot_results(input_path: Path, output_path: Path, dataset: str) -> None:
    import matplotlib.pyplot as plt

    means = _read_metric(input_path, "mean_ndcg")
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
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        positions,
        [value for _, value in ordered],
        color="#4C78A8",
        marker="o",
        linewidth=2,
    )
    for position, (_, value) in zip(positions, ordered):
        ax.annotate(
            f"{value:.3f}",
            (position, value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_title(dataset.upper())
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [labels[strategy] for strategy, _ in ordered], rotation=45, ha="right"
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean NDCG")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.suptitle("BEIR retrieval quality by model, tools, and SID reasoning effort")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_pareto_results(
    input_path: Path,
    output_path: Path,
    dataset: str,
    *,
    cost: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    if cost:
        resource_key = "estimated_token_cost"
        resource_label = "Estimated API token cost per query (USD; lower is better)"
        resource_values = _read_metric(
            input_path, "mean_resource.output_tokens", model_cost=True
        )
        title = "GPT-5-mini, GPT-6 Luna, and SID-2 estimated token-cost Pareto frontier"
    else:
        resource_key = "mean_resource.output_tokens"
        resource_label = "Mean reasoning + output tokens per query (lower is better)"
        resource_values = _read_metric(input_path, resource_key)
        title = "GPT-5-mini, GPT-6 Luna, and SID-2 output-token Pareto frontier"

    means = _read_metric(input_path, "mean_ndcg")
    labels = dict(STRATEGIES)
    points = [
        {
            "strategy": strategy,
            "mean_ndcg": means[(dataset, strategy)],
            resource_key: value,
        }
        for strategy, _ in STRATEGIES
        if strategy in STRATEGY_MODELS
        and (dataset, strategy) in means
        and (dataset, strategy) in resource_values
        for value in [resource_values[(dataset, strategy)]]
    ]
    if not points:
        raise ValueError(f"No Pareto metrics found for dataset={dataset!r}.")

    frontier = _pareto_front(points, resource_key)
    frontier_names = {str(point["strategy"]) for point in frontier}
    fig, ax = plt.subplots(figsize=(10, 6))
    for point in points:
        strategy = str(point["strategy"])
        resource_value = float(point[resource_key])
        ndcg = float(point["mean_ndcg"])
        is_frontier = strategy in frontier_names
        ax.scatter(
            resource_value,
            ndcg,
            color="#F58518" if is_frontier else "#9E9E9E",
            s=55,
            zorder=3 if is_frontier else 2,
        )
        label = (
            labels[strategy]
            .replace("GPT-5-mini", "GPT-5m")
            .replace("SID-2 · ", "SID ")
            .replace(" · ", " ")
            .replace("BM25 + E5", "BM25+E5")
        )
        ax.annotate(
            label,
            (resource_value, ndcg),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    if frontier:
        ax.plot(
            [float(point[resource_key]) for point in frontier],
            [float(point["mean_ndcg"]) for point in frontier],
            color="#F58518",
            linewidth=2,
            marker="o",
            zorder=4,
        )
    ax.set_title(dataset.upper())
    ax.set_xlabel(resource_label)
    ax.set_ylabel("Mean NDCG (higher is better)")
    ax.grid(linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the SciFact and TREC-COVID SID matrix."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    prefix = args.output_prefix.with_suffix("") if args.output_prefix.suffix else args.output_prefix
    for dataset in DATASETS:
        plot_results(
            args.input,
            prefix.parent / f"{prefix.name}_{dataset}.png",
            dataset,
        )
        plot_pareto_results(
            args.input,
            prefix.parent / f"{prefix.name}_pareto_{dataset}.png",
            dataset,
        )
        plot_pareto_results(
            args.input,
            prefix.parent / f"{prefix.name}_cost_pareto_{dataset}.png",
            dataset,
            cost=True,
        )


if __name__ == "__main__":
    main()
