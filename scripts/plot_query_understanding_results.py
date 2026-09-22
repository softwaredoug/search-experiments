#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

EXPECTED_STRATEGIES = {
    "ecom_qu_category_bm25_filtered_llm_single",
    "ecom_qu_category_bm25_boosted_llm_single",
    "ecom_qu_category_bm25_hierarchy_boosted_llm_single",
    "ecom_qu_category_hierarchy_bm25_filtered_llm_single",
    "ecom_qu_category_hierarchy_bm25_boosted_llm_single",
    "ecom_qu_category_hierarchy_bm25_hierarchy_boosted_llm_single",
    "ecom_qu_category_bm25_filtered_llm_multiple",
    "ecom_qu_category_bm25_boosted_llm_multiple",
    "ecom_qu_category_bm25_hierarchy_boosted_llm_multiple",
    "ecom_qu_category_hierarchy_bm25_filtered_llm_multiple",
    "ecom_qu_category_hierarchy_bm25_boosted_llm_multiple",
    "ecom_qu_category_hierarchy_bm25_hierarchy_boosted_llm_multiple",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("dataset") == "wands"]
    latest = {row["strategy_name"]: row for row in rows}
    missing = EXPECTED_STRATEGIES - latest.keys()
    if missing:
        raise ValueError(f"Missing WANDS results for: {', '.join(sorted(missing))}")
    return list(latest.values())


def _short_label(row: dict[str, str]) -> str:
    field = "hierarchy" if row["category_field"] == "category hierarchy" else "category"
    retrieval = row["retrieval_engine"].removeprefix("bm25_").replace("_", "-")
    enrichment = row["enrichment_engine"].removeprefix("llm_")
    return f"{field}\n{retrieval}\n{enrichment}"


def _plot(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metric_name = rows[0]["metric_name"]
    metric_key = metric_name.lower()
    means = [float(row[f"mean_{metric_key}"]) for row in rows]
    labels = [_short_label(row) for row in rows]
    colors = {
        "bm25_filtered": "#4C78A8",
        "bm25_boosted": "#F58518",
        "bm25_hierarchy_boosted": "#54A24B",
    }
    hatches = {"llm_single": "", "llm_multiple": "//"}

    fig, ax = plt.subplots(figsize=(15, 7))
    bars = []
    for index, row in enumerate(rows):
        bar = ax.bar(
            index,
            means[index],
            color=colors[row["retrieval_engine"]],
            hatch=hatches[row["enrichment_engine"]],
        )[0]
        bars.append(bar)
        ax.annotate(
            f"{means[index]:.3f}",
            (bar.get_x() + bar.get_width() / 2, means[index]),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_title("WANDS Query Understanding Experiments")
    ax.set_ylabel(f"Mean {metric_name}")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    retrieval_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color) for color in colors.values()
    ]
    enrichment_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatch)
        for hatch in hatches.values()
    ]
    retrieval_legend = ax.legend(
        retrieval_handles,
        [name.removeprefix("bm25_") for name in colors],
        title="Retrieval engine",
        loc="upper left",
    )
    ax.add_artist(retrieval_legend)
    ax.legend(
        enrichment_handles,
        [name.removeprefix("llm_") for name in hatches],
        title="Enrichment engine",
        loc="upper right",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot WANDS query-understanding results.")
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results_query_understanding.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/query_understanding_wands.png"),
    )
    args = parser.parse_args()
    _plot(_load_rows(args.results_csv), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
