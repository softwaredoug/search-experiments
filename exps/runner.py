import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from exps.datasets import DATASET_NAMES, get_dataset
from exps.runners.run import RunParams, run_benchmark
from exps.strategy_factory import create_strategy, load_strategy


def _report_metric(metric_name: str, metric_series, graded=None) -> None:
    metric_key = metric_name.lower()
    if metric_series.empty:
        print(f"No {metric_name} results to report.")
        return

    display_series = metric_series
    if (
        graded is not None
        and display_series.index.name == "query_id"
        and "query" in graded.columns
        and "query_id" in graded.columns
    ):
        query_map = (
            graded[["query", "query_id"]]
            .drop_duplicates()
            .set_index("query_id")["query"]
        )
        display_series = display_series.copy()
        display_series.index = display_series.index.map(query_map.get)
        display_series.index.name = "query"

    print(f"Per-query {metric_name}:")
    print(display_series.to_string())
    print("")
    print("Summary:")
    print(f"mean_{metric_key}={metric_series.mean():.4f}")
    print(f"median_{metric_key}={metric_series.median():.4f}")


def _write_summary_csv(path: str, *, dataset: str, result) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "dataset": dataset,
        "strategy_name": result.strategy_name,
        "strategy_params": json.dumps(result.strategy_params, sort_keys=True),
        "metric_name": result.metric_name,
        **result.summary,
    }

    file_exists = output_path.exists()
    fieldnames = None
    if file_exists:
        with output_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames

    if not fieldnames:
        fieldnames = list(row.keys())

    write_header = not file_exists or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _display_title(row: pd.Series) -> str:
    title = row.get("title", "")
    if isinstance(title, str) and title.strip():
        return title
    if title:
        return str(title)
    description = row.get("description", "")
    return description if isinstance(description, str) else str(description)


def _grade_column(judgments: pd.DataFrame) -> str | None:
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in judgments.columns:
            return col
    return None


def _show_most_relevant(
    *, query: str, judgments: pd.DataFrame, corpus: pd.DataFrame
) -> None:
    if "query" not in judgments.columns:
        return
    grade_col = _grade_column(judgments)
    if grade_col is None:
        return
    subset = judgments[judgments["query"] == query]
    if subset.empty:
        return
    scores = pd.to_numeric(subset[grade_col], errors="coerce")
    if scores.notna().any():
        top_idx = scores.idxmax()
        top_row = subset.loc[top_idx]
    else:
        top_row = subset.iloc[0]
    doc_id = top_row.get("doc_id")
    title = ""
    if doc_id is not None and "doc_id" in corpus.columns:
        match = corpus[corpus["doc_id"] == doc_id]
        if not match.empty:
            title = _display_title(match.iloc[0])
    label = title if title else str(doc_id) if doc_id is not None else ""
    print("Most relevant result:")
    print(f"{label}\t{grade_col}={top_row.get(grade_col)}")
    print("")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment strategies.")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Path to strategy YAML config.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="wands",
        help="Dataset to run against.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries to sample for evaluation.",
    )
    parser.add_argument(
        "--query",
        help="Optional query string to show ranked results.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to show for --query.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for indexing/search.",
    )
    parser.add_argument(
        "--binary-relevance",
        help=(
            "Comma-separated fields to treat as binary relevance "
            "(title, description, category)."
        ),
    )
    parser.add_argument(
        "--device",
        help="Embedding device override (e.g., mps, cpu).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cached strategy results.",
    )
    parser.add_argument(
        "--summary-csv",
        help="Write summary stats to CSV (appends if exists).",
    )
    args = parser.parse_args()

    if args.query:
        strategy_config, strategy_params, requires_bm25 = load_strategy(
            args.strategy, device=args.device
        )
        dataset = get_dataset(
            args.dataset, workers=args.workers, ensure_snowball=requires_bm25
        )
        corpus = dataset.corpus
        judgments = dataset.judgments
        strategy, _ = create_strategy(
            strategy_config,
            corpus=corpus,
            workers=args.workers,
            params=strategy_params,
            device=args.device,
            dataset=args.dataset,
        )
        _show_most_relevant(query=args.query, judgments=judgments, corpus=corpus)
        top_k, scores = strategy.search(args.query, k=args.k)
        results = corpus.iloc[top_k].copy()
        results["score"] = scores
        grade_col = _grade_column(judgments)
        if grade_col and "query" in judgments.columns and "doc_id" in judgments.columns:
            match = judgments[judgments["query"] == args.query]
            grade_map = dict(zip(match["doc_id"], match[grade_col]))
            if "doc_id" in results.columns:
                results["grade"] = results["doc_id"].map(grade_map)

        print("Query results:")
        header_cols = ["score"]
        if grade_col and "grade" in results.columns:
            header_cols.append("grade")
        header = "doc_id\t" + "\t".join(header_cols) + "\ttitle"
        print(header)
        for _, row in results.iterrows():
            doc_id = row.get("doc_id", "")
            title = _display_title(row)
            values = [f"{row.get('score', 0):.4f}"]
            if grade_col and "grade" in results.columns:
                values.append(str(row.get("grade", "")))
            print(f"{doc_id}\t" + "\t".join(values) + f"\t{title}")
        return

    params = RunParams(
        strategy_path=args.strategy,
        dataset=args.dataset,
        num_queries=args.num_queries,
        seed=args.seed,
        workers=args.workers,
        binary_relevance=args.binary_relevance,
        device=args.device,
        no_cache=args.no_cache,
    )
    result = run_benchmark(params)
    _report_metric(result.metric_name, result.metric_series, result.graded)
    if args.summary_csv:
        _write_summary_csv(args.summary_csv, dataset=args.dataset, result=result)


if __name__ == "__main__":
    main()
