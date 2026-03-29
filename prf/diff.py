import argparse

import pandas as pd

from cheat_at_search import wands_data
from cheat_at_search.search import ndcgs, run_strategy

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


def _print_ndcg_diff(
    ndcg_a: pd.Series,
    ndcg_b: pd.Series,
    judgments: pd.DataFrame,
    sort_by: str,
) -> None:
    query_text = _query_text_map(judgments)
    df = pd.DataFrame({"ndcg_a": ndcg_a, "ndcg_b": ndcg_b})
    df["diff"] = df["ndcg_a"] - df["ndcg_b"]
    df.index.name = "query_id"
    df["query"] = df.index.map(query_text.get)
    df = df.reset_index()
    df = df[df["diff"] != 0]

    if df.empty:
        print("No NDCG differences found.")
        return

    if sort_by == "query":
        df = df.sort_values(by=["query", "diff"], ascending=[True, False])
    else:
        df = df.sort_values(by=["diff", "query"], ascending=[False, True])

    print("Per-query NDCG differences (A - B):")
    print(df[["query_id", "query", "ndcg_a", "ndcg_b", "diff"]].to_string())
    print("")
    print("Summary:")
    print(f"mean_diff={df['diff'].mean():.4f}")
    print(f"median_diff={df['diff'].median():.4f}")
    print("")
    print("Strategy summaries:")
    print(f"mean_ndcg_a={df['ndcg_a'].mean():.4f}")
    print(f"median_ndcg_a={df['ndcg_a'].median():.4f}")
    print(f"mean_ndcg_b={df['ndcg_b'].mean():.4f}")
    print(f"median_ndcg_b={df['ndcg_b'].median():.4f}")


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
        "--sort",
        choices=["delta", "query"],
        default="delta",
        help="Sort per-query diff output.",
    )
    args = parser.parse_args()

    corpus = wands_data.corpus
    judgments = wands_data.judgments

    strategy_a = STRATEGIES[args.strategy_a](corpus)
    strategy_b = STRATEGIES[args.strategy_b](corpus)

    if args.query:
        print(f"Query: {args.query}")
        _print_query_results(
            "Strategy A", args.query, corpus, judgments, strategy_a, args.k
        )
        _print_query_results(
            "Strategy B", args.query, corpus, judgments, strategy_b, args.k
        )
        return

    graded_a = run_strategy(strategy_a, judgments)
    graded_b = run_strategy(strategy_b, judgments)
    ndcg_a = ndcgs(graded_a)
    ndcg_b = ndcgs(graded_b)
    _print_ndcg_diff(ndcg_a, ndcg_b, judgments, args.sort)


if __name__ == "__main__":
    main()
