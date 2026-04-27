#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


BASELINES = ["bm25", "bm25_strong_title", "embedding_minilm"]
DATASETS = ["esci", "wands"]


def _load_results(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _aggregate(rows: list[dict[str, str]]):
    agg = defaultdict(lambda: {"sum": 0.0, "count": 0})
    for row in rows:
        key = (row["dataset"], row["strategy_name"])
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
    ax.set_title(f"{dataset.upper()} NDCG (mean)")
    ax.set_ylabel("Mean NDCG")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylim(0, max(means) * 1.1 if means else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "results.csv"
    output_dir = repo_root / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_results(results_path)
    aggregated = _aggregate(rows)

    for dataset in DATASETS:
        output_path = output_dir / f"{dataset}_ndcg.png"
        _plot_dataset(aggregated, dataset, output_path)


if __name__ == "__main__":
    main()
