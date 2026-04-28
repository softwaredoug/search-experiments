#!/usr/bin/env python3
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


DATASET = "minimarco"
RESULTS_FILE = "results_minimarco.csv"
ORDER = [
    "bm25_msmarco",
    "embedding_e5_msmarco",
    "agentic_msmarco_bm25_gpt5_mini",
    "agentic_msmarco_bm25_e5_gpt5_mini",
    "agentic_msmarco_e5_gpt5_mini",
]


def _load_results(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _filtered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered = []
    for row in rows:
        if row.get("dataset") != DATASET:
            continue
        if row.get("metric_name", "").lower() != "mrr":
            continue
        filtered.append(row)
    return filtered


def _ordered_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_name = {row["strategy_name"]: row for row in rows}
    ordered = [by_name[name] for name in ORDER if name in by_name]
    missing = [name for name in ORDER if name not in by_name]
    if missing:
        raise ValueError(f"Missing minimarco rows for: {', '.join(missing)}")
    return ordered


def _plot(rows: list[dict[str, str]], output_path: Path) -> None:
    strategies = [row["strategy_name"] for row in rows]
    means = [float(row["mean_mrr"]) for row in rows]

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
    ax.set_title("MiniMSMARCO - MRR (mean)")
    ax.set_ylabel("Mean MRR")
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=30, ha="right")
    ax.set_ylim(0, max(means) * 1.1 if means else 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / RESULTS_FILE
    output_path = repo_root / "assets" / "minimarco_mrr.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _load_results(results_path)
    filtered = _filtered_rows(rows)
    ordered = _ordered_rows(filtered)
    _plot(ordered, output_path)


if __name__ == "__main__":
    main()
