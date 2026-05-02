#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASELINES = ["bm25", "bm25_strong_title", "embedding_e5"]
DATASETS = ["esci", "wands"]
GPT5_STRATEGY = "agentic_bm25_e5_ecommerce_gpt5"
ALLOW_AGENTIC = {"agentic_codegen"}
EXCLUDE_AGENTIC_MINILM = {
    "agentic_minilm_ecommerce_gpt5_mini",
    "agentic_bm25_minilm_ecommerce_gpt5_mini",
}
STRATEGY_ALIASES = {
    "agentic_bm25_e5_base_v2_ecommerce_gpt5_mini": "agentic_bm25_e5_ecommerce_gpt5_mini",
}
GPT5_HARDCODED_NDCG = {
    "esci": 0.4534540809414133,
    "wands": 0.6170634248596244,
}


def _load_results(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _strategy_model(row: dict[str, str]) -> str | None:
    try:
        params = json.loads(row.get("strategy_params", "{}"))
    except json.JSONDecodeError:
        return None
    model = params.get("model")
    return model if isinstance(model, str) else None


def _include_in_ndcg_plot(row: dict[str, str]) -> bool:
    strategy_name = row.get("strategy_name")
    if strategy_name in BASELINES:
        return True
    if strategy_name in ALLOW_AGENTIC:
        return True
    if strategy_name == GPT5_STRATEGY:
        return True
    model = _strategy_model(row)
    if model in {"gpt-5-mini", "gpt-5.1-mini"}:
        if strategy_name in EXCLUDE_AGENTIC_MINILM:
            return False
        if isinstance(strategy_name, str) and strategy_name.startswith("agentic_"):
            return True
    return False


def _aggregate(rows: list[dict[str, str]]):
    agg = defaultdict(lambda: {"sum": 0.0, "count": 0})
    for row in rows:
        strategy_name = STRATEGY_ALIASES.get(row["strategy_name"], row["strategy_name"])
        key = (row["dataset"], strategy_name)
        agg[key]["sum"] += float(row["mean_ndcg"])
        agg[key]["count"] += 1
    aggregated = []
    for (dataset, strategy), vals in agg.items():
        aggregated.append(
            {
                "dataset": dataset,
                "strategy": strategy,
                "mean": vals["sum"] / vals["count"],
            }
        )
    return aggregated


def _ordered_rows(rows: list[dict[str, str]]):
    baseline_rows = [row for row in rows if row["strategy"] in BASELINES]
    agentic_rows = [row for row in rows if row["strategy"] not in BASELINES]
    baseline_rows.sort(key=lambda row: BASELINES.index(row["strategy"]))
    agentic_rows.sort(key=lambda row: row["mean"])
    return baseline_rows + agentic_rows


def _plot_dataset(rows: list[dict[str, str]], dataset: str, output_path: Path) -> None:
    ordered = _ordered_rows([row for row in rows if row["dataset"] == dataset])
    strategies = [row["strategy"] for row in ordered]
    means = [row["mean"] for row in ordered]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(len(strategies)), means, color="#4C78A8", marker="o", linewidth=2)
    for idx, value in enumerate(means):
        ax.annotate(
            f"{value:.3f}",
            (idx, value),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=9,
        )
    if dataset == "esci":
        ax.set_title("ESCI - NDCG Mean (N=1000)")
    else:
        ax.set_title(f"{dataset.upper()} NDCG (mean)")
    ax.set_ylabel("Mean NDCG")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    y_min = 0.2 if dataset == "wands" else 0
    ax.set_ylim(y_min, max(means) * 1.1 if means else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    return re.sub(r"_+", "_", slug).strip("_") or "ndcg"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot agentic NDCG results.")
    parser.add_argument(
        "--name",
        help="Suffix label for the plot filename (slugified).",
        default="ndcg",
    )
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "results.csv"
    output_dir = repo_root / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [row for row in _load_results(results_path) if _include_in_ndcg_plot(row)]
    aggregated = _aggregate(rows)
    for dataset, mean_ndcg in GPT5_HARDCODED_NDCG.items():
        if any(
            row["dataset"] == dataset and row["strategy"] == GPT5_STRATEGY
            for row in aggregated
        ):
            continue
        aggregated.append(
            {
                "dataset": dataset,
                "strategy": GPT5_STRATEGY,
                "mean": mean_ndcg,
            }
        )

    suffix = _slugify(args.name)
    for dataset in DATASETS:
        output_path = output_dir / f"{dataset}_{suffix}.png"
        _plot_dataset(aggregated, dataset, output_path)


if __name__ == "__main__":
    main()
