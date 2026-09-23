#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot recall against coverage for Jev choice thresholds."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    points = [
        {
            "variant": row["variant"],
            "label": (
                "gpt-5-mini"
                if row["confidence_threshold"] == "baseline"
                else (
                    "gpt-5"
                    if row["confidence_threshold"] == "gpt-5"
                    else f"jev({row['confidence_threshold']})"
                )
            ),
            "coverage": float(row["coverage"]),
            "recall": float(row["recall"]),
        }
        for row in rows
        if row["recall"] not in ("", "None")
    ]
    points.sort(key=lambda point: point["coverage"])
    if not points:
        raise ValueError("No rows with recall and coverage were found in the input CSV.")

    coverages = [point["coverage"] for point in points]
    recalls = [point["recall"] for point in points]
    auc = sum(
        (right["coverage"] - left["coverage"])
        * (left["recall"] + right["recall"])
        / 2
        for left, right in zip(points, points[1:])
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(coverages, recalls, color="#4C78A8", marker="o", linewidth=2)
    for point in points:
        ax.annotate(
            point["label"],
            (point["coverage"], point["recall"]),
            textcoords="offset points",
            xytext=(12, -8),
            ha="right",
            va="top",
        )
    ax.set_title(f"Choice classifier: recall vs coverage (AUC={auc:.3f})")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Mean recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    plt.close(fig)
    print(f"AUC: {auc:.6f}")


if __name__ == "__main__":
    main()
