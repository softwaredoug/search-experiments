import pandas as pd

from cheat_at_search.search import run_strategy
from prf.cache import load_cached_results
from prf.datasets import bm25_params_for_dataset, get_dataset
from prf.metrics import metric_for_dataset
from prf.runners.diff import DiffParams, diff_benchmark
from prf.runners.run import RunParams, run_benchmark
from prf.strategy_config import load_strategy_config, resolve_strategy_class


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


def test_run_benchmark_wands_cache_roundtrip():
    params = RunParams(
        strategy_path="configs/bm25_strong_title.yml",
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        binary_relevance="title",
        device=None,
    )
    first = run_benchmark(params)

    strategy_config = load_strategy_config(params.strategy_path)
    strategy_params = dict(strategy_config.params)
    bm25_k1, bm25_b = bm25_params_for_dataset(params.dataset)
    if "bm25_k1" not in strategy_params and "k1" not in strategy_params:
        strategy_params["bm25_k1"] = bm25_k1
    if "bm25_b" not in strategy_params and "b" not in strategy_params:
        strategy_params["bm25_b"] = bm25_b

    dataset = get_dataset(params.dataset)
    judgments = dataset.judgments
    available_queries = judgments[["query", "query_id"]].drop_duplicates()
    if params.num_queries:
        available_queries = available_queries.sample(
            params.num_queries, random_state=params.seed
        )
    queries = available_queries["query"].tolist()
    query_list_hash = "|".join(map(str, queries))

    cached = load_cached_results(
        dataset=params.dataset,
        strategy_type=strategy_config.type,
        params=strategy_params,
        num_queries=params.num_queries,
        seed=params.seed,
        query_list_hash=query_list_hash,
    )
    assert cached is not None

    second = run_benchmark(params)
    pd.testing.assert_series_equal(first.metric_series, second.metric_series)


def test_run_benchmark_chunked_matches_direct():
    params = RunParams(
        strategy_path="configs/bm25_strong_title.yml",
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        binary_relevance=None,
        device=None,
        no_cache=True,
    )
    chunked = run_benchmark(params)

    strategy_config = load_strategy_config(params.strategy_path)
    strategy_cls = resolve_strategy_class(strategy_config.type)
    strategy_params = dict(strategy_config.params)
    bm25_k1, bm25_b = bm25_params_for_dataset(params.dataset)
    if "bm25_k1" not in strategy_params and "k1" not in strategy_params:
        strategy_params["bm25_k1"] = bm25_k1
    if "bm25_b" not in strategy_params and "b" not in strategy_params:
        strategy_params["bm25_b"] = bm25_b

    dataset = get_dataset(params.dataset)
    corpus = dataset.corpus
    judgments = dataset.judgments
    strategy = strategy_cls(corpus, workers=params.workers, **strategy_params)
    available_queries = judgments[["query", "query_id"]].drop_duplicates()
    available_queries = available_queries.sample(params.num_queries, random_state=params.seed)
    queries = available_queries["query"].tolist()
    direct_graded = run_strategy(strategy, judgments, queries=queries, seed=params.seed)
    metric_name, metric_fn = metric_for_dataset(params.dataset)
    direct_series = metric_fn(direct_graded)
    if direct_series.index.name != "query_id" and "query_id" in direct_graded.columns:
        query_map = (
            direct_graded[["query", "query_id"]]
            .drop_duplicates()
            .set_index("query")["query_id"]
        )
        direct_series = direct_series.copy()
        direct_series.index = direct_series.index.map(query_map.get)
        direct_series.index.name = "query_id"

    pd.testing.assert_series_equal(chunked.metric_series, direct_series)
