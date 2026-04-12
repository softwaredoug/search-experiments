import argparse

import pandas as pd
from cheat_at_search.search import run_strategy

from prf.datasets import (
    bm25_params_for_dataset,
    get_dataset,
    load_bm25_cache,
    save_bm25_cache,
)
from prf.metrics import metric_for_dataset
from prf.strategies.bm25 import BM25Strategy
from prf.strategies.doubleidf_bm25 import DoubleIDFBM25Strategy
from prf.strategies.reweighed_bm25 import ReweighedBM25Strategy
from prf.strategies.prf_rerank import PRFRerankStrategy

STRATEGIES = {
    "bm25": BM25Strategy,
    "bm25_doubleidf": DoubleIDFBM25Strategy,
    "bm25_reweighed": ReweighedBM25Strategy,
    "prf_rerank": PRFRerankStrategy,
}


def _report_metric(metric_name: str, metric_series: pd.Series) -> None:
    metric_key = metric_name.lower()
    if metric_series.empty:
        print(f"No {metric_name} results to report.")
        return

    print(f"Per-query {metric_name}:")
    print(metric_series.to_string())
    print("")
    print("Summary:")
    print(f"mean_{metric_key}={metric_series.mean():.4f}")
    print(f"median_{metric_key}={metric_series.median():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRF lexical strategies.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(STRATEGIES.keys()),
        help="Strategy to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=["esci", "msmarco", "wands"],
        default="wands",
        help="Dataset to run against.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries to sample for evaluation.",
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
            "Comma-separated fields to use binary relevance in PRF "
            "(title, description, category)."
        ),
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, workers=args.workers)
    corpus = dataset.corpus
    judgments = dataset.judgments
    bm25_k1, bm25_b = bm25_params_for_dataset(args.dataset)

    strategy_cls = STRATEGIES[args.strategy]
    graded = None
    if args.strategy == "bm25":
        graded = load_bm25_cache(args.dataset, args.num_queries, args.seed)
    if graded is None:
        if args.strategy == "prf_rerank":
            strategy = strategy_cls(
                corpus,
                workers=args.workers,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
                binary_relevance_fields=args.binary_relevance,
            )
        else:
            strategy = strategy_cls(
                corpus,
                workers=args.workers,
                bm25_k1=bm25_k1,
                bm25_b=bm25_b,
            )
        graded = run_strategy(
            strategy,
            judgments,
            num_queries=args.num_queries,
            seed=args.seed,
        )
        if args.strategy == "bm25":
            save_bm25_cache(args.dataset, args.num_queries, args.seed, graded)
    metric_name, metric_fn = metric_for_dataset(args.dataset)
    metric_series = metric_fn(graded)
    _report_metric(metric_name, metric_series)


if __name__ == "__main__":
    main()
