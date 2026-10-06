"""Runner e2e tests.

See docs/runner_tests_prd.md for requirements.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from cheat_at_search.search import run_strategy

from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.runners.diff import DiffParams, diff_benchmark
from exps.runners.run import RunParams, run_benchmark
from exps.strategy_config import load_strategy_config, resolve_strategy_class
from tests.utils.embedding_mocks import build_mock_embeddings


def _write_bm25_config(tmp_path):
    config_path = tmp_path / "bm25.yml"
    config_path.write_text(
        """
strategy:
  name: bm25_fixture
  type: bm25
  params:
    k1: 1.2
    b: 0.75
    title_boost: 9.3
    description_boost: 4.1
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def _write_embedding_config(tmp_path):
    config_path = tmp_path / "embedding_e5_base_v2.yml"
    config_path.write_text(
        """
strategy:
  name: embedding_e5_fixture
  type: embedding
  params:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    query_prefix: "query: "
    document_prefix: "passage: "
""".lstrip(),
        encoding="utf-8",
    )
    return config_path


def test_run_benchmark_wands_bm25_all_params(tmp_path):
    config_path = _write_bm25_config(tmp_path)
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=2,
        seed=123,
        workers=1,
        binary_relevance="title",
        device=None,
    )
    result = run_benchmark(params)

    assert result.strategy_name == "bm25_fixture"
    assert isinstance(result.metric_series, pd.Series)
    assert result.metric_series.index.name == "query"
    assert not result.metric_series.empty
    assert "mean_" in next(iter(result.summary.keys()))
    assert result.summary["tool_calls_mean"] == 1.0
    assert result.summary["tool_calls_median"] == 1.0
    assert result.summary["tool_calls_std"] == 0.0


def test_diff_benchmark_wands_bm25_all_params(tmp_path):
    config_path = _write_bm25_config(tmp_path)
    params = DiffParams(
        strategy_a_path=str(config_path),
        strategy_b_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
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


def test_diff_benchmark_wands_query_results(tmp_path):
    config_path = _write_bm25_config(tmp_path)
    params = DiffParams(
        strategy_a_path=str(config_path),
        strategy_b_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        query="bm25",
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


def test_run_benchmark_query_results(tmp_path):
    config_path = _write_bm25_config(tmp_path)
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        query="bm25",
        k=5,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.query_results is not None
    assert len(result.query_results) == 5
    assert "score" in result.query_results.columns
    assert "display_title" in result.query_results.columns
    assert result.most_relevant_row is not None
    assert result.most_relevant_grade_col is not None
    assert result.relevant_examples is not None
    assert len(result.relevant_examples) <= 3
    for example in result.relevant_examples:
        assert "doc_id" in example
        assert "title" in example
        assert "description" in example


def test_run_benchmark_matches_direct(tmp_path):
    config_path = _write_bm25_config(tmp_path)
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=2,
        seed=123,
        workers=1,
        binary_relevance=None,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    strategy_config = load_strategy_config(params.strategy_path, base_path=params.base_path)
    strategy_cls = resolve_strategy_class(strategy_config.type)
    strategy_params = dict(strategy_config.params)
    dataset = get_dataset(params.dataset)
    corpus = dataset.corpus
    judgments = dataset.judgments
    strategy = strategy_cls(corpus, workers=params.workers, **strategy_params)
    available_queries = judgments[["query", "query_id"]].drop_duplicates()
    available_queries = available_queries.sample(params.num_queries, random_state=params.seed)
    queries = available_queries["query"].tolist()
    direct_graded = run_strategy(
        strategy,
        judgments,
        queries=queries,
        seed=params.seed,
        cache=not params.no_cache,
    )
    metric_name, metric_fn = metric_for_dataset(params.dataset)
    direct_series = metric_fn(direct_graded)
    pd.testing.assert_series_equal(result.metric_series, direct_series)


@patch("exps.runners.run.get_dataset")
@patch("cheat_at_search.embeddings._cache_root")
@patch("exps.strategies.embedding.load_or_create_embeddings")
@patch("exps.strategies.embedding.load_model")
def test_run_benchmark_embedding_prefixes(
    mock_load_model,
    mock_load_or_create_embeddings,
    mock_cache_root,
    mock_get_dataset,
    tmp_path,
    doug_blog_dataset,
):
    def fake_cache_root():
        return tmp_path

    corpus = doug_blog_dataset.corpus
    judgments = doug_blog_dataset.judgments
    dataset = SimpleNamespace(corpus=corpus, judgments=judgments)
    mock_cache_root.side_effect = fake_cache_root
    mock_get_dataset.return_value = dataset
    model_holder: dict[str, object] = {}

    def _mock_load_or_create_embeddings(corpus, passage_fn, **_kwargs):
        embeddings, model = build_mock_embeddings(
            corpus,
            judgments,
            passage_fn,
            dim=3,
            seed=123,
        )
        model_holder["model"] = model
        return embeddings, model

    mock_load_or_create_embeddings.side_effect = _mock_load_or_create_embeddings
    mock_load_model.side_effect = lambda *_args, **_kwargs: model_holder.get("model")

    config_path = _write_embedding_config(tmp_path)
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert len(result.metric_series) == 1
