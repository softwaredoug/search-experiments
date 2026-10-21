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
    ("agentic_bm25_ecommerce", "GPT-5 · BM25"),
    ("agentic_bm25_e5_ecommerce_gpt5", "GPT-5 · BM25 + E5"),
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
GPT5_STRATEGIES = frozenset(
    {
        "agentic_bm25_ecommerce",
        "agentic_bm25_e5_ecommerce_gpt5",
    }
)
PARETO_STRATEGIES = tuple(
    strategy for strategy in STRATEGIES if strategy[0] not in GPT5_STRATEGIES
)
TOKEN_PRICES_PER_MILLION = {
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.0},
    "gpt-6-luna": {"input": 0.1, "cached_input": 0.01, "output": 0.5},
    "sid-2": {"input": 0.25, "cached_input": 0.025, "output": 1.0},
}
TOKEN_PRICING_MODELS = {
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


def _read_mean_ndcg(
    path: Path,
    *,
    allow_missing: bool = False,
) -> dict[tuple[str, str], float]:
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
    if missing and not allow_missing:
        missing_text = ", ".join(f"{dataset}/{strategy}" for dataset, strategy in missing)
        raise ValueError(f"No mean NDCG results found for: {missing_text}")
    return means


def _read_mean_output_tokens(path: Path) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    strategy_names = {name for name, _ in STRATEGIES}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "dataset",
            "strategy_name",
            "metric_name",
            "mean_resource.output_tokens",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {', '.join(sorted(required))} columns.")
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            output_tokens = row["mean_resource.output_tokens"]
            if (
                dataset not in DATASETS
                or strategy not in strategy_names
                or row["metric_name"].lower() != "ndcg"
                or not output_tokens
            ):
                continue
            try:
                totals[(dataset, strategy)].append(float(output_tokens))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mean_resource.output_tokens for dataset={dataset!r}, "
                    f"strategy={strategy!r}."
                ) from exc

    return {
        key: sum(values) / len(values)
        for key, values in totals.items()
        if values
    }


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
            model = TOKEN_PRICING_MODELS.get(strategy)
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
            cost = (
                (input_count - cached_count) * prices["input"]
                + cached_count * prices["cached_input"]
                + output_count * prices["output"]
            ) / 1_000_000
            totals[(dataset, strategy)].append(cost)

    return {
        key: sum(values) / len(values)
        for key, values in totals.items()
        if values
    }


def _pareto_front(
    points: list[dict[str, float | str]],
    resource_key: str = "mean_resource.output_tokens",
) -> list[dict[str, float | str]]:
    """Return points not dominated on retrieval (higher) and resource cost (lower)."""
    frontier = []
    for point in points:
        ndcg = float(point["mean_ndcg"])
        resource_cost = float(point[resource_key])
        dominated = any(
            float(other["mean_ndcg"]) >= ndcg
            and float(other[resource_key]) <= resource_cost
            and (
                float(other["mean_ndcg"]) > ndcg
                or float(other[resource_key]) < resource_cost
            )
            for other in points
            if other is not point
        )
        if not dominated:
            frontier.append(point)
    return sorted(
        frontier, key=lambda point: float(point[resource_key])
    )


def _avoid_label_overlaps(annotations, ax, fig) -> None:
    """Move Pareto labels vertically until each is readable and inside its axes."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_bounds = ax.get_window_extent(renderer)
    placed = []
    padding = fig.dpi * 4 / 72
    offsets = [0, *(offset for step in range(1, 10) for offset in (step * 16, -step * 16))]
    for annotation in annotations:
        x, original_y = annotation.get_position()
        for offset in offsets:
            annotation.set_position((x, original_y + offset))
            bounds = annotation.get_window_extent(renderer)
            inside_axes = bounds.y0 >= axes_bounds.y0 and bounds.y1 <= axes_bounds.y1
            padded_bounds = bounds.padded(padding)
            if inside_axes and not any(
                padded_bounds.overlaps(previous) for previous in placed
            ):
                placed.append(padded_bounds)
                break


def plot_results(input_path: Path, output_path: Path, dataset: str) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path, allow_missing=True)
    max_ndcg_by_dataset = read_mean_max_ndcg(input_path)
    labels = dict(STRATEGIES)
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {DATASETS}.")
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

    fig, ax = plt.subplots(figsize=(13, 6))
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
    ax.set_xticklabels([labels[strategy] for strategy, _ in ordered], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean NDCG")
    if dataset in max_ndcg_by_dataset:
        add_mean_max_ndcg_line(ax, max_ndcg_by_dataset[dataset])
        ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.suptitle("E-commerce retrieval quality by model, tools, and SID-2 reasoning effort")
    missing = [
        labels[strategy]
        for strategy, _ in STRATEGIES
        if (dataset, strategy) not in means
    ]
    if missing:
        fig.text(0.5, 0.01, f"Missing results omitted: {', '.join(missing)}", ha="center")
    fig.tight_layout(rect=(0, 0.04 if missing else 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_pareto_points(
    points: list[dict[str, float | str]],
    output_path: Path,
    dataset: str,
    *,
    resource_key: str,
    resource_label: str,
    title: str,
    excluded: list[str],
) -> None:
    import matplotlib.pyplot as plt

    if not points:
        raise ValueError(f"No NDCG and {resource_label} results found for {dataset!r}.")
    pareto_labels = {
        strategy: (
            label.replace("GPT-5-mini", "GPT-5m")
            .replace("SID-2 · ", "SID ")
            .replace(" · ", " ")
            .replace("BM25 + E5", "BM25+E5")
        )
        for strategy, label in STRATEGIES
    }
    frontier = _pareto_front(points, resource_key)
    frontier_names = {str(point["strategy"]) for point in frontier}
    fig, ax = plt.subplots(figsize=(10, 6))
    annotations = []
    for point in points:
        ndcg = float(point["mean_ndcg"])
        resource_value = float(point[resource_key])
        is_frontier = str(point["strategy"]) in frontier_names
        ax.scatter(
            resource_value,
            ndcg,
            color="#F58518" if is_frontier else "#9E9E9E",
            s=55,
            zorder=3 if is_frontier else 2,
        )
        annotations.append(
            ax.annotate(
                pareto_labels[str(point["strategy"])],
                (resource_value, ndcg),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                fontsize=8,
            )
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
    notes = [f"Excluded: {', '.join(excluded)}"] if excluded else []
    if notes:
        fig.text(0.5, 0.01, "; ".join(notes), ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05 if notes else 0, 1, 0.96))
    _avoid_label_overlaps(annotations, ax, fig)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_pareto_points(
    input_path: Path,
    dataset: str,
    resource_values: dict[tuple[str, str], float],
    resource_key: str,
) -> tuple[list[dict[str, float | str]], list[str]]:
    means = _read_mean_ndcg(input_path, allow_missing=True)
    labels = dict(STRATEGIES)
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; expected one of {DATASETS}.")
    points = [
        {
            "strategy": strategy,
            "label": labels[strategy],
            "mean_ndcg": means[(dataset, strategy)],
            resource_key: resource_values[(dataset, strategy)],
        }
        for strategy, _ in PARETO_STRATEGIES
        if (dataset, strategy) in resource_values and (dataset, strategy) in means
    ]
    excluded = [
        labels[strategy]
        for strategy, _ in STRATEGIES
        if (dataset, strategy) not in means or (dataset, strategy) not in resource_values
    ]
    return points, excluded


def plot_pareto_results(input_path: Path, output_path: Path, dataset: str) -> None:
    output_tokens = _read_mean_output_tokens(input_path)
    points, excluded = _build_pareto_points(
        input_path, dataset, output_tokens, "mean_resource.output_tokens"
    )
    _plot_pareto_points(
        points,
        output_path,
        dataset,
        resource_key="mean_resource.output_tokens",
        resource_label="Mean Reasoning + Output Tokens per Query (lower is better)",
        title="GPT-5-mini, GPT-6 Luna, and SID-2 output-token Pareto frontier",
        excluded=excluded,
    )


def plot_cost_pareto_results(input_path: Path, output_path: Path, dataset: str) -> None:
    token_cost_usd = _read_mean_token_cost_usd(input_path)
    points, excluded = _build_pareto_points(
        input_path, dataset, token_cost_usd, "mean_token_cost_usd"
    )
    _plot_pareto_points(
        points,
        output_path,
        dataset,
        resource_key="mean_token_cost_usd",
        resource_label="Estimated API Token Cost per Query (USD; lower is better)",
        title="GPT-5-mini, GPT-6 Luna, and SID-2 estimated token-cost Pareto frontier",
        excluded=excluded,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot e-commerce SID matrix retrieval quality, output-token Pareto, "
            "and estimated token-cost Pareto."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--pareto-input",
        type=Path,
        help="Optional CSV containing the metrics used for Pareto plots.",
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    pareto_input = args.pareto_input or args.input
    prefix = args.output_prefix
    if prefix.suffix:
        prefix = prefix.with_suffix("")
    for dataset in DATASETS:
        retrieval_output = prefix.parent / f"{prefix.name}_{dataset}.png"
        pareto_output = prefix.parent / f"{prefix.name}_pareto_{dataset}.png"
        cost_pareto_output = prefix.parent / f"{prefix.name}_cost_pareto_{dataset}.png"
        plot_results(args.input, retrieval_output, dataset)
        plot_pareto_results(pareto_input, pareto_output, dataset)
        plot_cost_pareto_results(pareto_input, cost_pareto_output, dataset)
        print(f"Wrote {retrieval_output}")
        print(f"Wrote {pareto_output}")
        print(f"Wrote {cost_pareto_output}")


if __name__ == "__main__":
    main()
