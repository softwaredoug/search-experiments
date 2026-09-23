#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot accuracy against coverage for Jev choice thresholds."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    with args.input.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    baseline_labels = {
        "baseline": "gpt-5-mini",
        "gpt-5": "gpt-5",
        "gpt-6-sol": "gpt-6-sol",
        "gpt-6-luna": "gpt-6-luna",
    }
    points = []
    for row in rows:
        if row["recall"] in ("", "None"):
            continue
        threshold = row["confidence_threshold"]
        points.append(
            {
                "variant": row["variant"],
                "label": baseline_labels.get(threshold, f"jev({threshold})"),
                "coverage": float(row["coverage"]),
                "recall": float(row["recall"]),
            }
        )
    points.sort(key=lambda point: point["coverage"])
    if not points:
        raise ValueError("No rows with recall and coverage were found in the input CSV.")

    # Alternate labels above and below the curve in any cluster whose adjacent
    # coverage values are within five percentage points of one another.
    label_offsets = [None] * len(points)
    close_x_gap = 0.05
    group_start = 0
    for group_end in range(1, len(points) + 1):
        if (
            group_end < len(points)
            and points[group_end]["coverage"] - points[group_end - 1]["coverage"]
            <= close_x_gap
        ):
            continue
        if group_end - group_start > 1:
            for index in range(group_start, group_end):
                group_index = index - group_start
                above = group_index % 2 == 0
                lane = group_index // 2
                vertical_offset = 10 + lane * 18
                label_offsets[index] = (
                    0,
                    vertical_offset if above else -vertical_offset,
                )
        group_start = group_end

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
    for index, point in enumerate(points):
        offset = label_offsets[index]
        ax.annotate(
            point["label"],
            (point["coverage"], point["recall"]),
            textcoords="offset points",
            xytext=offset if offset is not None else (12, -8),
            ha="center" if offset is not None else "right",
            va=("bottom" if offset[1] > 0 else "top") if offset is not None else "top",
            arrowprops=(
                {
                    "arrowstyle": "-",
                    "linestyle": ":",
                    "color": "#666666",
                    "linewidth": 0.8,
                    "shrinkA": 3,
                    "shrinkB": 5,
                }
                if offset is not None
                else None
            ),
        )
    ax.set_title("Choice classifier: accuracy vs coverage")
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
