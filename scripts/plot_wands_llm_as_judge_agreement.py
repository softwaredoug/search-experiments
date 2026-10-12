#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

STRATEGIES = (
    ("wands_pointwise_judge_gpt5_mini", "GPT-5-mini pointwise"),
    ("wands_pointwise_judge_gpt5", "GPT-5 pointwise"),
    ("wands_pointwise_judge_jev_latest", "Jev latest pointwise"),
    ("wands_pairwise_judge_gpt5_mini", "GPT-5-mini pairwise"),
    ("wands_pairwise_judge_jev_latest", "Jev latest pairwise"),
)
AGREEMENT_FIELDS = ("mean_agreement", "mean_pairwise_agreement")


def _read_agreement_means(path: Path) -> dict[tuple[str, str], float]:
    totals: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "strategy_name", "metric_name", *AGREEMENT_FIELDS}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain WANDS strategy and agreement summary columns."
            )
        for row in reader:
            dataset = row["dataset"]
            strategy = row["strategy_name"]
            if dataset != "wands" or strategy not in {name for name, _ in STRATEGIES}:
                raise ValueError(
                    f"Expected only WANDS judge rows; got {dataset}/{strategy}."
                )
            if row["metric_name"].lower() != "ndcg":
                raise ValueError(
                    f"Expected NDCG as the ranking metric for {strategy}, "
                    f"got {row['metric_name']!r}."
                )
            for field in AGREEMENT_FIELDS:
                try:
                    value = float(row[field])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid {field} for WANDS/{strategy}."
                    ) from exc
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"{field} must be between 0 and 1 for {strategy}.")
                totals[(strategy, field)].append(value)

    means = {key: sum(values) / len(values) for key, values in totals.items() if values}
    missing = [
        f"{strategy}/{field}"
        for strategy, _ in STRATEGIES
        for field in AGREEMENT_FIELDS
        if (strategy, field) not in means
    ]
    if missing:
        raise ValueError("Missing WANDS agreement results for: " + ", ".join(missing))
    return means


def plot_agreement(input_path: Path, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    means = _read_agreement_means(input_path)
    labels = [label for _, label in STRATEGIES]
    positions = list(range(len(STRATEGIES)))
    width = 0.34
    fig, axis = plt.subplots(figsize=(12, 5))

    for index, (field, label, color) in enumerate(
        (
            ("mean_agreement", "Pointwise agreement", "#4C78A8"),
            ("mean_pairwise_agreement", "Pairwise agreement", "#F58518"),
        )
    ):
        values = [means[(strategy, field)] for strategy, _ in STRATEGIES]
        offsets = [position + (index - 0.5) * width for position in positions]
        bars = axis.bar(offsets, values, width=width, label=label, color=color)
        for bar, value in zip(bars, values):
            axis.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean agreement")
    axis.set_title("WANDS judge agreement")
    axis.grid(axis="y", linestyle="--", alpha=0.4)
    axis.set_axisbelow(True)
    axis.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot pointwise and pairwise judge agreement on WANDS."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot_agreement(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
