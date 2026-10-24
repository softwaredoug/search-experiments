import pandas as pd

from cheat_at_search.search import run_strategy
from cheat_at_search.strategy import SearchStrategy


def test_cached_search_results_receive_both_ndcg_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "cheat_at_search.strategy.strategy.ensure_data_subdir",
        lambda _: tmp_path,
    )
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "title": ["low grade", "high grade"],
            "description": ["text", "text"],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": [1, 1],
            "query": ["test query", "test query"],
            "doc_id": [1, 2],
            "grade": [1, 4],
        }
    )

    class CountingStrategy(SearchStrategy):
        @property
        def cache_key(self):
            return "ndcg-post-search-cache-test"

        def __init__(self):
            super().__init__(corpus)
            self.calls = 0

        def search(self, query, k=10):
            self.calls += 1
            return [0], [self.calls]

    strategy = CountingStrategy()
    first = run_strategy(
        strategy, judgments, seed=17, cache=True, show_progress=False
    )
    cached = run_strategy(
        strategy, judgments, seed=17, cache=True, show_progress=False
    )

    assert strategy.calls == 1
    assert {"ndcg_g", "ndcg_m", "max_ndcg_m"}.issubset(cached.columns)
    assert cached["ndcg_g"].iloc[0] != cached["ndcg_m"].iloc[0]
    pd.testing.assert_series_equal(first["ndcg_g"], cached["ndcg_g"])
    pd.testing.assert_series_equal(first["ndcg_m"], cached["ndcg_m"])
