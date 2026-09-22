#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

EXPECTED_STRATEGIES = {
    "ecom_qu_category_bm25_filtered_llm_single",
    "ecom_qu_category_hierarchy_bm25_filtered_llm_single",
    "ecom_qu_category_bm25_filtered_llm_multiple",
    "ecom_qu_category_hierarchy_bm25_filtered_llm_multiple",
}
EXPECTED_NDCG_STRATEGIES = {"bm25"} | {
    strategy
    for field in ("category", "category_hierarchy")
    for retrieval in ("bm25_filtered", "bm25_boosted", "bm25_hierarchy_boosted")
    for enrichment in ("llm_single", "llm_multiple")
    for strategy in [f"ecom_qu_{field}_{retrieval}_{enrichment}"]
}


def _normalise_strategy_name(strategy: str) -> str:
    if strategy in EXPECTED_STRATEGIES:
        return strategy
    config_name = Path(strategy).stem
    prefix = "ecom_query_understanding_"
    if config_name.startswith(prefix):
        return "ecom_qu_" + config_name.removeprefix(prefix)
    return strategy


def _load_rows(path: Path, eval_as: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == "wands" and row.get("eval_as") == eval_as
        ]
    latest = {}
    for row in rows:
        row = dict(row)
        row["strategy"] = _normalise_strategy_name(row["strategy"])
        latest[row["strategy"]] = row
    missing = EXPECTED_STRATEGIES - latest.keys()
    if missing:
        raise ValueError(f"Missing WANDS results for: {', '.join(sorted(missing))}")
    return sorted(
        latest.values(),
        key=lambda row: float(row["mean_recall"] or 0),
    )


def _load_ndcg_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("dataset") == "wands"
            and row.get("metric_name", "").lower() == "ndcg"
        ]
    latest = {row["strategy_name"]: row for row in rows}
    missing = EXPECTED_NDCG_STRATEGIES - latest.keys()
    if missing:
        raise ValueError(f"Missing WANDS NDCG results for: {', '.join(sorted(missing))}")
    return sorted(
        latest.values(),
        key=lambda row: float(row["mean_ndcg"] or 0),
    )


def _short_label(strategy: str) -> str:
    if strategy == "bm25":
        return "BM25\nbaseline"
    field = "hierarchy" if "_category_hierarchy_" in strategy else "category"
    retrieval = next(
        retrieval
        for retrieval in ("bm25_filtered", "bm25_boosted", "bm25_hierarchy_boosted")
        if f"_{retrieval}_" in strategy
    )
    retrieval = retrieval.removeprefix("bm25_").replace("_", "-")
    enrichment = "multiple" if strategy.endswith("llm_multiple") else "single"
    return f"{field}\n{retrieval}\n{enrichment}"


def _plot(rows: list[dict[str, str]], eval_as: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [_short_label(row["strategy"]) for row in rows]
    recalls = [float(row["mean_recall"] or 0) for row in rows]
    jaccards = [float(row["mean_jaccard"] or 0) for row in rows]
    x_values = range(len(rows))

    fig, ax = plt.subplots(figsize=(15, 7))
    for values, label, color in (
        (recalls, "Mean recall", "#4C78A8"),
        (jaccards, "Mean Jaccard", "#F58518"),
    ):
        ax.plot(x_values, values, color=color, marker="o", linewidth=2, label=label)
        for index, value in enumerate(values):
            ax.annotate(
                f"{value:.3f}",
                (index, value),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=9,
            )
    ax.set_title(f"WANDS Query Understanding Classification: {eval_as}")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_xticks(list(x_values))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_ndcg(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [_short_label(row["strategy_name"]) for row in rows]
    ndcgs = [float(row["mean_ndcg"] or 0) for row in rows]
    x_values = range(len(rows))

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(x_values, ndcgs, marker="o", linewidth=2)
    for index, value in enumerate(ndcgs):
        ax.annotate(
            f"{value:.3f}",
            (index, value),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=9,
        )
    ax.set_title("WANDS Query Understanding Retrieval")
    ax.set_ylabel("Mean NDCG")
    ax.set_ylim(min(ndcgs) - 0.05, max(ndcgs) + 0.05)
    ax.set_xticks(list(x_values))
    ax.set_xticklabels(labels, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    results_dir = Path(__file__).resolve().parents[1] / "research" / "results"
    parser = argparse.ArgumentParser(
        description="Plot WANDS query-understanding classification recall."
    )
    parser.add_argument(
        "--eval-as",
        required=True,
        choices=("taxonomy[0]", "taxonomy[1]", "direct"),
        help="Classification perspective to plot.",
    )
    parser.add_argument(
        "--classification-csv",
        type=Path,
        default=results_dir / "query_understanding_classification_wands.csv",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=results_dir / "query_understanding_retrieval.csv",
    )
    parser.add_argument(
        "--ndcg-output",
        type=Path,
        default=Path("assets/query_understanding_ndcg_wands.png"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    eval_as_filename = args.eval_as.replace("[", "_").replace("]", "")
    output_path = args.output or Path(
        f"assets/query_understanding_classification_wands_{eval_as_filename}.png"
    )
    rows = _load_rows(args.classification_csv, args.eval_as)
    _plot(rows, args.eval_as, output_path)
    _plot_ndcg(_load_ndcg_rows(args.results_csv), args.ndcg_output)
    print(f"Wrote {output_path}")
    print(f"Wrote {args.ndcg_output}")


if __name__ == "__main__":
    main()
