import pandas as pd

from prf.runners.diff import DiffParams, diff_benchmark
from prf.runners.run import RunParams, run_benchmark


def test_run_benchmark_wands_bm25_all_params():
    params = RunParams(
        strategy_path="configs/bm25_strong_title.yml",
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        binary_relevance="title",
        device=None,
    )
    result = run_benchmark(params)

    assert result.strategy_name == "bm25_strong_title"
    assert isinstance(result.metric_series, pd.Series)
    assert result.metric_series.index.name == "query_id"
    assert not result.metric_series.empty
    assert "mean_" in next(iter(result.summary.keys()))


def test_diff_benchmark_wands_bm25_all_params():
    params = DiffParams(
        strategy_a_path="configs/bm25_strong_title.yml",
        strategy_b_path="configs/bm25_strong_title.yml",
        dataset="wands",
        query=None,
        k=5,
        num_queries=2,
        seed=123,
        workers=1,
        sort="delta",
        binary_relevance="title",
        device=None,
    )
    result = diff_benchmark(params)

    assert isinstance(result.metric_a, pd.Series)
    assert isinstance(result.metric_b, pd.Series)
    assert "diff" in result.diff_table.columns


def test_diff_benchmark_wands_query_results():
    params = DiffParams(
        strategy_a_path="configs/bm25_strong_title.yml",
        strategy_b_path="configs/bm25_strong_title.yml",
        dataset="wands",
        query="salon chair",
        k=5,
        num_queries=1,
        seed=123,
        workers=1,
        sort="delta",
        binary_relevance="title",
        device=None,
    )
    result = diff_benchmark(params)

    assert result.query_results_a is not None
    assert result.query_results_b is not None
    assert len(result.query_results_a) == 5
    assert len(result.query_results_b) == 5
