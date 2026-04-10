import argparse

import numpy as np
import pandas as pd
from cheat_at_search.search import ndcgs, run_strategy

from prf.datasets import get_dataset
from prf.runner import STRATEGIES


def _grade_column(judgments: pd.DataFrame) -> str | None:
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in judgments.columns:
            return col
    return None


def _query_text_map(judgments: pd.DataFrame) -> dict:
    if "query_id" in judgments.columns and "query" in judgments.columns:
        return (
            judgments[["query_id", "query"]]
            .drop_duplicates()
            .set_index("query_id")["query"]
            .to_dict()
        )
    return {}


def _ndcg_for_query(strategy, judgments: pd.DataFrame, query: str) -> float | None:
    if "query" not in judgments.columns:
        return None
    subset = judgments[judgments["query"] == query]
    if subset.empty:
        return None
    graded = run_strategy(strategy, subset)
    series = ndcgs(graded)
    if series.empty:
        return None
    return float(series.iloc[0])


def _print_query_results(
    label: str,
    query: str,
    corpus: pd.DataFrame,
    judgments: pd.DataFrame,
    strategy,
    k: int,
) -> None:
    top_k, scores = strategy.search(query, k=k)
    results = corpus.iloc[top_k].copy()
    results["score"] = scores

    grade_col = _grade_column(judgments)
    grades = {}
    if grade_col and "query" in judgments.columns and "doc_id" in judgments.columns:
        match = judgments[judgments["query"] == query]
        grades = dict(zip(match["doc_id"], match[grade_col]))

    print("")
    print(f"{label} results:")
    for _, row in results.iterrows():
        doc_id = row.get("doc_id", "")
        title = row.get("title", "")
        score = row.get("score", 0)
        grade = grades.get(doc_id, "")
        print(f"{doc_id}\t{score:.4f}\t{grade}\t{title}")
    ndcg_value = _ndcg_for_query(strategy, judgments, query)
    if ndcg_value is None:
        print("NDCG: unavailable")
    else:
        print(f"NDCG: {ndcg_value:.4f}")


def _print_ndcg_diff(
    ndcg_a: pd.Series,
    ndcg_b: pd.Series,
    judgments: pd.DataFrame,
    sort_by: str,
    name_a: str,
    name_b: str,
) -> None:
    query_text = _query_text_map(judgments)
    col_a = f"ndcg_{name_a}"
    col_b = f"ndcg_{name_b}"
    df_all = pd.DataFrame({col_a: ndcg_a, col_b: ndcg_b})
    df_all["diff"] = df_all[col_b] - df_all[col_a]
    df_all.index.name = "query_id"
    df_all["query"] = df_all.index.map(query_text.get)
    df_all = df_all.reset_index()

    df = df_all[df_all["diff"] != 0]

    if df.empty:
        print("No NDCG differences found.")
        return

    if sort_by == "query":
        df = df.sort_values(by=["query", "diff"], ascending=[True, False])
    else:
        df = df.sort_values(by=["diff", "query"], ascending=[False, True])

    print(f"Per-query NDCG differences ({name_b} - {name_a}):")
    print(df[["query_id", "query", col_a, col_b, "diff"]].to_string())
    print("")
    print("Summary:")
    print(f"mean_diff={df['diff'].mean():.4f}")
    print(f"median_diff={df['diff'].median():.4f}")
    print("")
    print("Strategy summaries:")
    print(f"mean_{col_a}={df_all[col_a].mean():.4f}")
    print(f"median_{col_a}={df_all[col_a].median():.4f}")
    print(f"mean_{col_b}={df_all[col_b].mean():.4f}")
    print(f"median_{col_b}={df_all[col_b].median():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two strategies.")
    parser.add_argument(
        "--strategy-a",
        required=True,
        choices=sorted(STRATEGIES.keys()),
        help="First strategy to compare.",
    )
    parser.add_argument(
        "--strategy-b",
        required=True,
        choices=sorted(STRATEGIES.keys()),
        help="Second strategy to compare.",
    )
    parser.add_argument(
        "--dataset",
        choices=["esci", "msmarco", "wands"],
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
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    dataset = get_dataset(args.dataset)
    corpus = dataset.corpus
    judgments = dataset.judgments

    strategy_a = STRATEGIES[args.strategy_a](corpus, workers=args.workers)
    strategy_b = STRATEGIES[args.strategy_b](corpus, workers=args.workers)

    if args.query:
        print(f"Query: {args.query}")
        _print_query_results(
            f"{args.strategy_a}",
            args.query,
            corpus,
            judgments,
            strategy_a,
            args.k,
        )
        _print_query_results(
            f"{args.strategy_b}",
            args.query,
            corpus,
            judgments,
            strategy_b,
            args.k,
        )
        return

    graded_a = run_strategy(
        strategy_a,
        judgments,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    graded_b = run_strategy(
        strategy_b,
        judgments,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    ndcg_a = ndcgs(graded_a)
    ndcg_b = ndcgs(graded_b)
    _print_ndcg_diff(
        ndcg_a,
        ndcg_b,
        judgments,
        args.sort,
        args.strategy_a,
        args.strategy_b,
    )


if __name__ == "__main__":
    main()
