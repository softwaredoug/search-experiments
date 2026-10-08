#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

DATASETS = ("wands", "esci")
STRATEGIES = (
    ("bm25", "BM25"),
    ("bag_of_decisions_direct", "Jev Reranker"),
    ("bag_of_decisions_example", "Bag of Decisions"),
)


def _read_mean_ndcg(path: Path) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "strategy_name", "metric_name", "mean_ndcg"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain {', '.join(sorted(required))} columns."
            )
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            if (
                dataset not in DATASETS
                or strategy not in {name for name, _ in STRATEGIES}
            ):
                continue
            if row["metric_name"].lower() != "ndcg":
                raise ValueError(
                    f"Expected NDCG for {dataset}/{strategy}, got {row['metric_name']!r}."
                )
            try:
                values[(dataset, strategy)].append(float(row["mean_ndcg"]))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid mean_ndcg for dataset={dataset!r}, strategy={strategy!r}."
                ) from exc

    means = {}
    duplicates = []
    for key, scores in values.items():
        if len(scores) != 1:
            duplicates.append(f"{key[0]}/{key[1]} ({len(scores)} rows)")
        else:
            means[key] = scores[0]
    if duplicates:
        raise ValueError("Expected one result row per dataset/variant: " + ", ".join(duplicates))

    missing = [
        f"{dataset}/{strategy}"
        for dataset in DATASETS
        for strategy, _ in STRATEGIES
        if (dataset, strategy) not in means
    ]
    if missing:
        raise ValueError("Missing NDCG results for: " + ", ".join(missing))
    return means


def plot_results(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = _read_mean_ndcg(input_path)
    labels = [label for _, label in STRATEGIES]
    positions = list(range(len(STRATEGIES)))
    colors = {"wands": "#4C78A8", "esci": "#F58518"}

    fig, axis = plt.subplots(figsize=(9, 5))
    for dataset in DATASETS:
        scores = [means[(dataset, strategy)] for strategy, _ in STRATEGIES]
        axis.plot(
            positions,
            scores,
            marker="o",
            linewidth=2,
            color=colors[dataset],
            label=dataset.upper(),
        )
        for position, score in zip(positions, scores):
            axis.annotate(
                f"{score:.3f}",
                (position, score),
                xytext=(0, 8 if dataset == "wands" else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if dataset == "wands" else "top",
                fontsize=9,
                color=colors[dataset],
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean NDCG")
    axis.set_title("Bag of Decisions reranker beats just Jev reranker")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    axis.legend(title="Dataset")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot BM25 and bag-of-decisions mean NDCG for WANDS and ESCI."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_results(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
