from cheat_at_search import wands_data
from cheat_at_search.search import ndcgs, run_strategy

from prf.strategies.bm25 import BM25Strategy


def test_bm25_wands_ndcg_sanity():
    corpus = wands_data.corpus
    judgments = wands_data.judgments

    strategy = BM25Strategy(corpus)
    graded = run_strategy(strategy, judgments, num_queries=5, seed=42, cache=True)
    ndcg_series = ndcgs(graded)

    assert len(graded) > 0
    assert not ndcg_series.empty
    assert ndcg_series.mean() > 0
