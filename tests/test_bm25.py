from cheat_at_search import wands_data
from cheat_at_search.search import ndcgs, run_strategy

from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.strategies.bm25 import BM25Strategy


def test_bm25_wands_ndcg_sanity():
    corpus = wands_data.corpus
    judgments = wands_data.judgments

    strategy = BM25Strategy(corpus)
    graded = run_strategy(strategy, judgments, num_queries=5, seed=42, cache=True)
    ndcg_series = ndcgs(graded)

    assert len(graded) > 0
    assert not ndcg_series.empty
    assert ndcg_series.mean() > 0


def test_bm25_minimarco_mrr_sanity():
    dataset = get_dataset("minimarco", ensure_snowball=False)
    corpus = dataset.corpus
    judgments = dataset.judgments

    strategy = BM25Strategy(corpus)
    graded = run_strategy(strategy, judgments, num_queries=5, seed=42, cache=True)
    metric_name, metric_fn = metric_for_dataset("minimarco")
    metric_series = metric_fn(graded)

    assert metric_name == "MRR"
    assert len(graded) > 0
    assert not metric_series.empty
    assert (metric_series >= 0).all()
    assert (metric_series <= 1).all()
