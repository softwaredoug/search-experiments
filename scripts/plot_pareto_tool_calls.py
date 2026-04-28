#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from exps.paths import AGENTIC_TRACE_ROOT
from exps.trace_utils import slugify


DATASETS = ["esci", "wands"]
GPT5_STRATEGY = "agentic_bm25_e5_ecommerce_gpt5"
PARETO_STRATEGIES = {
    "agentic_e5_ecommerce_gpt5_mini",
    "agentic_bm25_ecommerce_gpt5_mini",
    "agentic_bm25_e5_base_v2_ecommerce_gpt5_mini",
}

LABEL_MAP = {
    "agentic_e5_ecommerce_gpt5_mini": "embedding",
    "agentic_bm25_ecommerce_gpt5_mini": "bm25",
    "agentic_bm25_minilm_ecommerce_gpt5_mini": "bm25+embedding",
    "agentic_bm25_e5_base_v2_ecommerce_gpt5_mini": "bm25+embeddings",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _strategy_model(row: dict[str, str]) -> str | None:
    try:
        params = json.loads(row.get("strategy_params", "{}"))
    except json.JSONDecodeError:
        return None
    model = params.get("model")
    return model if isinstance(model, str) else None


def _include_in_pareto(row: dict[str, str]) -> bool:
    strategy_name = row.get("strategy_name")
    if strategy_name in PARETO_STRATEGIES:
        return True
    if strategy_name == GPT5_STRATEGY:
        return True
    return False


def _load_ndcg(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "count": 0.0})
    for row in rows:
        if row.get("metric_name", "").lower() != "ndcg":
            continue
        if not _include_in_pareto(row):
            continue
        dataset = row.get("dataset")
        strategy = row.get("strategy_name")
        if not dataset or not strategy:
            continue
        key = (dataset, strategy)
        agg[key]["sum"] += float(row.get("mean_ndcg", 0.0))
        agg[key]["count"] += 1.0

    ndcg = {}
    for key, vals in agg.items():
        count = vals["count"] or 1.0
        ndcg[key] = vals["sum"] / count
    return ndcg


def _load_tool_calls(rows: list[dict[str, str]]) -> dict[tuple[str, str], float]:
    agg: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "count": 0.0})
    for row in rows:
        if row.get("metric_name", "").lower() != "ndcg":
            continue
        if not _include_in_pareto(row):
            continue
        dataset = row.get("dataset")
        strategy = row.get("strategy_name")
        if not dataset or not strategy:
            continue
        tool_calls = row.get("tool_calls_mean")
        if tool_calls is None or tool_calls == "":
            continue
        key = (dataset, strategy)
        agg[key]["sum"] += float(tool_calls)
        agg[key]["count"] += 1.0
    tool_calls = {}
    for key, vals in agg.items():
        count = vals["count"] or 1.0
        tool_calls[key] = vals["sum"] / count
    return tool_calls


def _latest_trace_dir(strategy_dir: Path) -> Path | None:
    timestamp_dirs = [path for path in strategy_dir.iterdir() if path.is_dir()]
    if not timestamp_dirs:
        return None
    return sorted(timestamp_dirs, key=lambda path: path.name)[-1]


def _tool_calls_from_traces() -> dict[tuple[str, str], float]:
    tool_calls = {}
    if not AGENTIC_TRACE_ROOT.exists():
        return tool_calls
    for dataset_dir in AGENTIC_TRACE_ROOT.iterdir():
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for strategy_dir in dataset_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name
            latest_dir = _latest_trace_dir(strategy_dir)
            if latest_dir is None:
                continue
            counts = []
            for query_dir in latest_dir.iterdir():
                if not query_dir.is_dir():
                    continue
                summary_path = query_dir / "summary.json"
                if not summary_path.exists():
                    continue
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    continue
                num_calls = summary.get("num_tool_calls")
                if isinstance(num_calls, (int, float)):
                    counts.append(float(num_calls))
            if counts:
                tool_calls[(dataset, strategy)] = sum(counts) / len(counts)
    return tool_calls


def _pareto_front(points: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    ordered = sorted(points, key=lambda row: (row["tool_calls_mean"], -row["mean_ndcg"]))
    frontier = []
    best_ndcg = -1.0
    for row in ordered:
        ndcg = float(row["mean_ndcg"])
        if ndcg > best_ndcg:
            frontier.append(row)
            best_ndcg = ndcg
    return frontier


def _plot_dataset(rows: list[dict[str, float | str]], dataset: str, output_path: Path) -> None:
    points = [row for row in rows if row["dataset"] == dataset]
    if not points:
        return

    frontier = _pareto_front(points)
    if not frontier:
        return

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x_vals = [row["tool_calls_mean"] for row in frontier]
    y_vals = [row["mean_ndcg"] for row in frontier]
    ax.scatter(x_vals, y_vals, color="#F58518", s=60, zorder=2)

    for row in frontier:
        label = LABEL_MAP.get(str(row["strategy"]), str(row["strategy"]))
        ax.annotate(
            label,
            (row["tool_calls_mean"], row["mean_ndcg"]),
            textcoords="offset points",
            xytext=(0, -12),
            ha="center",
            va="top",
            fontsize=8,
        )

    fx = [row["tool_calls_mean"] for row in frontier]
    fy = [row["mean_ndcg"] for row in frontier]
    ax.plot(fx, fy, color="#F58518", linewidth=2, marker="o", zorder=3)

    ax.set_title(f"{dataset.upper()} - GPT-5-mini Pareto (Tool Calls vs NDCG)")
    ax.set_xlabel("Mean Tool Calls")
    ax.set_ylabel("Mean NDCG")
    if dataset == "esci":
        ax.set_ylim(0.2, 0.5)
    if dataset == "wands":
        ax.set_ylim(0.5, 0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "results.csv"
    output_dir = repo_root / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(results_path)
    ndcg = _load_ndcg(rows)
    tool_calls = _tool_calls_from_traces()
    fallback_tool_calls = _load_tool_calls(rows)
    aggregated = []
    for (dataset, strategy), mean_ndcg in ndcg.items():
        tool_calls_mean = tool_calls.get((dataset, slugify(strategy, fallback="strategy")))
        if tool_calls_mean is None:
            tool_calls_mean = fallback_tool_calls.get((dataset, strategy))
        if tool_calls_mean is None:
            continue
        aggregated.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "mean_ndcg": mean_ndcg,
                "tool_calls_mean": tool_calls_mean,
            }
        )
    for dataset in DATASETS:
        output_path = output_dir / f"{dataset}_pareto_gpt5_mini.png"
        _plot_dataset(aggregated, dataset, output_path)


if __name__ == "__main__":
    main()
