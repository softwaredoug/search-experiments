#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DATASETS = ("esci", "wands")
STRATEGIES = (
    ("bm25", "BM25"),
    ("embedding_e5", "E5-base-v2"),
    ("agentic_bm25_e5_ecommerce_gpt5_mini", "Agentic BM25 + E5"),
    ("agentic_ecom_composite_rrf_gpt5_mini", "BM25 + E5 RRF"),
    ("agentic_ecom_composite_rrf_jev_gpt5_mini", "BM25 + E5 RRF + Jev"),
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


def plot_results(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path)
    labels = [label for _, label in STRATEGIES]
    positions = range(len(STRATEGIES))
    colors = ["#4C78A8", "#72B7B2", "#F2CF5B", "#54A24B", "#E45756"]

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(15, 5), sharey=True)
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
    fig.suptitle("E-commerce retrieval tool variants")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot e-commerce tool variant NDCG.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
