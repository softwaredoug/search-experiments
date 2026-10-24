from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def is_ndcg_m_row(row: dict[str, str]) -> bool:
    """Accept the current metric label and legacy NDCG summary rows."""
    return row.get("metric_name", "").lower() in {"ndcg_m", "ndcg"}


def mean_ndcg_m_value(row: dict[str, str]) -> str | None:
    """Read the renamed field, falling back to historical CSVs."""
    return row.get("mean_ndcg_m") or row.get("mean_ndcg")


def mean_max_ndcg_m_value(row: dict[str, str]) -> str | None:
    """Read the renamed judged ceiling, falling back to historical CSVs."""
    return row.get("mean_max_ndcg_m") or row.get("mean_max_ndcg")


def mean_max_ndcg_m_by_dataset(rows: Iterable[dict[str, str]]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not is_ndcg_m_row(row):
            continue
        dataset = row.get("dataset", "")
        raw_value = mean_max_ndcg_m_value(row)
        if not dataset or not raw_value:
            continue
        try:
            values[dataset].append(float(raw_value))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid mean_max_ndcg_m for dataset={dataset!r}."
            ) from exc
    return {
        dataset: sum(dataset_values) / len(dataset_values)
        for dataset, dataset_values in values.items()
    }


def read_mean_max_ndcg_m(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or not {"dataset", "metric_name"}.issubset(reader.fieldnames)
            or not {"mean_max_ndcg_m", "mean_max_ndcg"}.intersection(
                reader.fieldnames
            )
        ):
            return {}
        return mean_max_ndcg_m_by_dataset(reader)


def add_mean_max_ndcg_m_line(ax, value: float) -> None:
    ax.axhline(
        value,
        color="#777777",
        linestyle=":",
        linewidth=1.5,
        label=f"Mean max NDCG_M (judged ceiling): {value:.3f}",
        zorder=4,
    )
