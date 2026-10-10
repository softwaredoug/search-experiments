#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from exps.paths import SEARCH_EXPERIMENTS_ROOT, slugify

DATASETS = ("esci", "wands")
STRATEGIES = (
    ("bm25", "BM25"),
    ("embedding_e5", "E5-base-v2"),
    ("agentic_bm25_e5_ecommerce_gpt5_mini", "Agentic BM25 + E5"),
    ("agentic_ecom_bm25_fewshot_judge_jev", "Few-shot + Jev judge"),
    ("agentic_ecom_bm25_fewshot_judge", "Few-shot + GPT-5-mini judge"),
)
AGENTIC_STRATEGIES = {
    "agentic_bm25_e5_ecommerce_gpt5_mini",
    "agentic_ecom_bm25_fewshot_judge_jev",
    "agentic_ecom_bm25_fewshot_judge",
}
TOOL_CALL_FIELDS = (
    "dataset",
    "strategy_name",
    "query",
    "query_slug",
    "num_tool_calls",
    "trace_timestamp",
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


def _query_tool_calls(
    *,
    dataset: str,
    strategy: str,
    trace_root: Path | None = None,
) -> list[dict[str, str | int | float]]:
    """Read the latest available trace count for each query.

    Empty trace runs are skipped, so cached strategy runs can reuse the most
    recent prior trace for each query.
    """
    root = trace_root or (SEARCH_EXPERIMENTS_ROOT / "agentic")
    strategy_dir = root / slugify(dataset, fallback="dataset") / slugify(
        strategy, fallback="strategy"
    )
    if not strategy_dir.is_dir():
        return []

    latest_by_query: dict[str, dict[str, str | int | float]] = {}
    timestamp_dirs = sorted(
        (path for path in strategy_dir.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    for timestamp_dir in timestamp_dirs:
        for query_dir in sorted(timestamp_dir.iterdir(), key=lambda path: path.name):
            if not query_dir.is_dir() or query_dir.name in latest_by_query:
                continue
            summary_path = query_dir / "summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            num_tool_calls = summary.get("num_tool_calls")
            if (
                not isinstance(num_tool_calls, (int, float))
                or isinstance(num_tool_calls, bool)
            ):
                continue
            query = summary.get("query")
            latest_by_query[query_dir.name] = {
                "dataset": dataset,
                "strategy_name": strategy,
                "query": query if isinstance(query, str) and query else query_dir.name,
                "query_slug": query_dir.name,
                "num_tool_calls": num_tool_calls,
                "trace_timestamp": timestamp_dir.name,
            }
    return list(latest_by_query.values())


def _collect_tool_calls(
    summary_path: Path,
    *,
    trace_root: Path | None = None,
) -> list[dict[str, str | int | float]]:
    pairs = set()
    with summary_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            pair = (row.get("dataset"), row.get("strategy_name"))
            if pair[1] in AGENTIC_STRATEGIES and pair[0]:
                pairs.add(pair)
    rows = []
    for dataset, strategy in sorted(pairs):
        rows.extend(
            _query_tool_calls(
                dataset=dataset,
                strategy=strategy,
                trace_root=trace_root,
            )
        )
    return rows


def _write_tool_calls(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TOOL_CALL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_pareto_points(
    summary_path: Path,
    tool_calls_path: Path | None,
) -> dict[str, list[dict[str, float | str]]]:
    ndcg_totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    summary_calls: dict[tuple[str, str], list[float]] = defaultdict(list)
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
            raw_calls = row.get("tool_calls_mean")
            if raw_calls not in (None, ""):
                try:
                    summary_calls[(dataset, strategy)].append(float(raw_calls))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid tool_calls_mean for {dataset}/{strategy}."
                    ) from exc

    traced_calls: dict[tuple[str, str], list[float]] = defaultdict(list)
    if tool_calls_path is not None and tool_calls_path.is_file():
        with tool_calls_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    traced_calls[(row["dataset"], row["strategy_name"])].append(
                        float(row["num_tool_calls"])
                    )
                except (KeyError, TypeError, ValueError):
                    continue

    labels = dict(STRATEGIES)
    points_by_dataset: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for (dataset, strategy), ndcg_values in ndcg_totals.items():
        calls = traced_calls.get((dataset, strategy)) or summary_calls.get(
            (dataset, strategy), []
        )
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
                "tool_calls_mean": mean_calls,
            }
        )
    return points_by_dataset


def _pareto_front(
    points: list[dict[str, float | str]],
) -> list[dict[str, float | str]]:
    ordered = sorted(
        points,
        key=lambda row: (float(row["tool_calls_mean"]), -float(row["mean_ndcg"])),
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
    tool_calls_path: Path | None,
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    points_by_dataset = _read_pareto_points(summary_path, tool_calls_path)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(14, 5), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        points = points_by_dataset.get(dataset, [])
        frontier = _pareto_front(points)
        frontier_names = {str(point["strategy"]) for point in frontier}
        for point in points:
            is_frontier = str(point["strategy"]) in frontier_names
            ax.scatter(
                point["tool_calls_mean"],
                point["mean_ndcg"],
                color="#F58518" if is_frontier else "#9E9E9E",
                s=65,
                zorder=3 if is_frontier else 2,
            )
            ax.annotate(
                str(point["label"]),
                (point["tool_calls_mean"], point["mean_ndcg"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )
        if frontier:
            ax.plot(
                [point["tool_calls_mean"] for point in frontier],
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
    parser.add_argument("--tool-calls-output", type=Path)
    parser.add_argument("--pareto-output", type=Path)
    args = parser.parse_args()
    if args.tool_calls_output is not None:
        _write_tool_calls(args.tool_calls_output, _collect_tool_calls(args.input))
    plot_results(args.input, args.output)
    if args.pareto_output is not None:
        plot_pareto_results(args.input, args.tool_calls_output, args.pareto_output)
    print(f"Wrote {args.output}")
    if args.tool_calls_output is not None:
        print(f"Wrote {args.tool_calls_output}")
    if args.pareto_output is not None:
        print(f"Wrote {args.pareto_output}")


if __name__ == "__main__":
    main()
