import pandas as pd

from cheat_at_search import wands_data
from cheat_at_search.search import ndcgs, run_strategy

from prf.strategies.bm25 import BM25Strategy


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
    corpus = wands_data.corpus
    judgments = wands_data.judgments

    strategy = BM25Strategy(corpus)
    graded = run_strategy(strategy, judgments)
    ndcg_series = ndcgs(graded)
    _report_ndcgs(ndcg_series)


if __name__ == "__main__":
    main()
