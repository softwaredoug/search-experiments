from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def mean_max_ndcg_by_dataset(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("metric_name", "").lower() != "ndcg":
            continue
        dataset = row.get("dataset", "")
        raw_value = row.get("mean_max_ndcg")
        if not dataset or not raw_value:
            continue
        try:
            values[dataset].append(float(raw_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid mean_max_ndcg for dataset={dataset!r}."
            ) from exc
    return {
        dataset: sum(dataset_values) / len(dataset_values)
        for dataset, dataset_values in values.items()
    }


def read_mean_max_ndcg(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {
            "dataset",
            "metric_name",
            "mean_max_ndcg",
        }.issubset(reader.fieldnames):
            return {}
        return mean_max_ndcg_by_dataset(reader)


def add_mean_max_ndcg_line(ax, value: float) -> None:
    ax.axhline(
        value,
        color="#777777",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean max NDCG (judged ceiling): {value:.3f}",
        zorder=4,
    )
