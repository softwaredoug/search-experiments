import argparse

import pandas as pd

from cheat_at_search import wands_data
from cheat_at_search.search import ndcgs, run_strategy

from prf.strategies.bm25 import BM25Strategy
from prf.strategies.prf import PRFStrategy

STRATEGIES = {
    "bm25": BM25Strategy,
    "prf": PRFStrategy,
}


def _report_ndcgs(ndcg_series: pd.Series) -> None:
    if ndcg_series.empty:
        print("No NDCG results to report.")
        return

    print("Per-query NDCG:")
    print(ndcg_series.to_string())
    print("")
    print("Summary:")
    print(f"mean_ndcg={ndcg_series.mean():.4f}")
    print(f"median_ndcg={ndcg_series.median():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRF lexical strategies.")
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(STRATEGIES.keys()),
        help="Strategy to run.",
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
    args = parser.parse_args()

    corpus = wands_data.corpus
    judgments = wands_data.judgments

    strategy_cls = STRATEGIES[args.strategy]
    strategy = strategy_cls(corpus)
    graded = run_strategy(
        strategy,
        judgments,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    ndcg_series = ndcgs(graded)
    _report_ndcgs(ndcg_series)


if __name__ == "__main__":
    main()
