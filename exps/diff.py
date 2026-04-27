import argparse
from datetime import datetime

import pandas as pd

from exps.datasets import DATASET_NAMES, get_dataset
from exps.runners.diff import DiffParams, diff_benchmark
from exps.strategy_factory import create_strategy, load_strategy
from exps.trace_utils import build_agentic_trace_root


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


def _print_query_results(label: str, results: pd.DataFrame) -> None:
    print("")
    print(f"{label} results:")
    for _, row in results.iterrows():
        doc_id = row.get("doc_id", "")
        title = _display_title(row)
        score = row.get("score", 0)
        grade = row.get("grade", "")
        print(f"{doc_id}\t{score:.4f}\t{grade}\t{title}")


def _print_metric_diff(
    metric_name: str,
    diff_table: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> None:
    if diff_table.empty:
        print(f"No {metric_name} differences found.")
        return

    metric_key = metric_name.lower()
    col_a = f"{metric_key}_{name_a}"
    col_b = f"{metric_key}_{name_b}"
    print(f"Per-query {metric_name} differences ({name_b} - {name_a}):")
    print(diff_table[["query_id", "query", col_a, col_b, "diff"]].to_string())
    print("")
    print("Summary:")
    print(f"mean_diff={diff_table['diff'].mean():.4f}")
    print(f"median_diff={diff_table['diff'].median():.4f}")
    print("")
    print("Strategy summaries:")
    print(f"mean_{col_a}={diff_table[col_a].mean():.4f}")
    print(f"median_{col_a}={diff_table[col_a].median():.4f}")
    print(f"mean_{col_b}={diff_table[col_b].mean():.4f}")
    print(f"median_{col_b}={diff_table[col_b].median():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two strategies.")
    parser.add_argument(
        "--strategy-a",
        required=True,
        help="Path to first strategy YAML config.",
    )
    parser.add_argument(
        "--strategy-b",
        required=True,
        help="Path to second strategy YAML config.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="wands",
        help="Dataset to run against.",
    )
    parser.add_argument(
        "--query",
        help="Optional query string to inspect.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to show for single-query diff.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries to sample for comparison.",
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
        "--sort",
        choices=["delta", "query"],
        default="delta",
        help="Sort per-query diff output.",
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
        help=(
            "Bypass cached strategy results (run_strategy only; does not affect "
            "BM25 indices or embeddings)."
        ),
    )
    args = parser.parse_args()

    if args.query:
        run_started_at = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        strategy_a_config, params_a, requires_bm25_a = load_strategy(
            args.strategy_a, device=args.device
        )
        strategy_b_config, params_b, requires_bm25_b = load_strategy(
            args.strategy_b, device=args.device
        )
        requires_bm25 = requires_bm25_a or requires_bm25_b
        dataset = get_dataset(
            args.dataset, workers=args.workers, ensure_snowball=requires_bm25
        )
        corpus = dataset.corpus
        judgments = dataset.judgments
        trace_path_a = None
        if strategy_a_config.type == "agentic":
            trace_path_a = build_agentic_trace_root(
                strategy_a_config.name,
                args.dataset,
                run_started_at=run_started_at,
            )
        trace_path_b = None
        if strategy_b_config.type == "agentic":
            trace_path_b = build_agentic_trace_root(
                strategy_b_config.name,
                args.dataset,
                run_started_at=run_started_at,
            )
        strategy_a, _ = create_strategy(
            strategy_a_config,
            corpus=corpus,
            workers=args.workers,
            params=params_a,
            device=args.device,
            dataset=args.dataset,
            trace_path=trace_path_a,
        )
        strategy_b, _ = create_strategy(
            strategy_b_config,
            corpus=corpus,
            workers=args.workers,
            params=params_b,
            device=args.device,
            dataset=args.dataset,
            trace_path=trace_path_b,
        )
        _show_most_relevant(query=args.query, judgments=judgments, corpus=corpus)
        query_results_a = _query_results(
            strategy_a, corpus, judgments, args.query, args.k
        )
        query_results_b = _query_results(
            strategy_b, corpus, judgments, args.query, args.k
        )
        _print_query_results(strategy_a_config.name, query_results_a)
        _print_query_results(strategy_b_config.name, query_results_b)
        return

    params = DiffParams(
        strategy_a_path=args.strategy_a,
        strategy_b_path=args.strategy_b,
        dataset=args.dataset,
        query=args.query,
        k=args.k,
        num_queries=args.num_queries,
        seed=args.seed,
        workers=args.workers,
        sort=args.sort,
        binary_relevance=args.binary_relevance,
        device=args.device,
        no_cache=args.no_cache,
    )
    result = diff_benchmark(params)

    if args.query:
        print(f"Query: {args.query}")
        if result.query_results_a is not None:
            _print_query_results(result.strategy_a_name, result.query_results_a)
            if result.query_metric_a is None:
                print(f"{result.metric_name}: unavailable")
            else:
                print(f"{result.metric_name}: {result.query_metric_a:.4f}")
        if result.query_results_b is not None:
            _print_query_results(result.strategy_b_name, result.query_results_b)
            if result.query_metric_b is None:
                print(f"{result.metric_name}: unavailable")
            else:
                print(f"{result.metric_name}: {result.query_metric_b:.4f}")
        return

    _print_metric_diff(
        result.metric_name,
        result.diff_table,
        result.strategy_a_name,
        result.strategy_b_name,
    )


if __name__ == "__main__":
    main()
