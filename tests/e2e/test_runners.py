"""Runner e2e tests.

See docs/runner_tests_prd.md for requirements.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import numpy as np
import pytest

from cheat_at_search.search import run_strategy

from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.runners.query_classification import (
    QueryClassificationParams,
    evaluate_query_classification,
)
from exps.runners.diff import DiffParams, diff_benchmark
from exps.runners.run import RunParams, run_benchmark
from exps.strategy_config import load_strategy_config, resolve_strategy_class
from exps.query_understanding import QueryUnderstandingStrategy
from exps.query_understanding.enrichers import make_dummy_enricher
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


def test_run_benchmark_query_understanding_dummy_doug_blog(
    tmp_path, doug_blog_dataset
):
    corpus = doug_blog_dataset.corpus.copy()
    vocabulary = [f"category-{index}" for index in range(6)]
    rng = np.random.default_rng(123)
    corpus["category"] = rng.choice(vocabulary, size=len(corpus))
    dataset = SimpleNamespace(corpus=corpus, judgments=doug_blog_dataset.judgments)

    config_path = tmp_path / "query_understanding.yml"
    config_path.write_text(
        """
strategy:
  name: query_understanding_dummy_fixture
  type: query_understanding
  params:
    categorize:
      field: category
      enrichment_engine:
        type: dummy
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title^9.4, description^4]
        boost_matches: 10
""".lstrip(),
        encoding="utf-8",
    )

    strategy = QueryUnderstandingStrategy(
        corpus,
        categorize={
            "field": "category",
        },
        retrieval_engine={
            "base": "bm25_boosted",
            "params": {
                "fields": ["title^9.4", "description^4"],
                "boost_matches": 10,
            },
        },
        enricher=make_dummy_enricher(vocabulary),
    )
    assert strategy.enricher.enrich("any query") == [vocabulary[0]]

    params = RunParams(
        strategy_path=str(config_path),
        dataset="doug_blog",
        num_queries=2,
        seed=123,
        workers=1,
        no_cache=True,
    )
    with patch("exps.runners.run.get_dataset", return_value=dataset):
        result = run_benchmark(params)

    assert result.strategy_name == "query_understanding_dummy_fixture"
    assert result.metric_name == "NDCG"
    assert result.metric_series is not None
    assert len(result.metric_series) == 2


def test_query_classification_backend_dummy_doug_blog(tmp_path, doug_blog_dataset):
    corpus = doug_blog_dataset.corpus.copy()
    vocabulary = [f"category-{index}" for index in range(6)]
    rng = np.random.default_rng(123)
    corpus["category"] = rng.choice(vocabulary, size=len(corpus))
    dataset = SimpleNamespace(corpus=corpus, judgments=doug_blog_dataset.judgments)

    config_path = tmp_path / "query_understanding.yml"
    config_path.write_text(
        """
strategy:
  name: query_understanding_dummy_fixture
  type: query_understanding
  params:
    categorize:
      field: category
      enrichment_engine:
        type: dummy
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title^9.4, description^4]
        boost_matches: 10
""".lstrip(),
        encoding="utf-8",
    )

    params = QueryClassificationParams(
        strategy_path=str(config_path),
        dataset="doug_blog",
        query_threshold=0.8,
    )
    with patch(
        "exps.runners.query_classification.get_dataset", return_value=dataset
    ):
        result = evaluate_query_classification(params)

    assert not result.per_query.empty
    assert set(result.per_query) == {
        "query",
        "expected_categories",
        "generated_categories",
        "recall",
        "jaccard",
    }
    assert result.per_query["generated_categories"].map(bool).all()
    assert 0.0 <= result.mean_recall <= 1.0
    assert 0.0 <= result.mean_jaccard <= 1.0
    assert result.coverage == 1.0

    unknown_params = QueryClassificationParams(
        strategy_path=str(config_path),
        dataset="doug_blog",
        query="query not present in judgments",
        query_threshold=0.8,
    )
    with patch(
        "exps.runners.query_classification.get_dataset", return_value=dataset
    ):
        unknown_result = evaluate_query_classification(unknown_params)

    unknown_row = unknown_result.per_query.iloc[0]
    assert unknown_row["expected_categories"] == []
    assert unknown_row["recall"] == 0.0
    assert unknown_row["jaccard"] == 0.0
    assert unknown_result.mean_recall == 0.0
    assert unknown_result.mean_jaccard == 0.0
    assert unknown_result.coverage == 1.0

    limited_params = QueryClassificationParams(
        strategy_path=str(config_path),
        dataset="doug_blog",
        limit=2,
        query_threshold=0.8,
    )
    with patch(
        "exps.runners.query_classification.get_dataset", return_value=dataset
    ):
        limited_result = evaluate_query_classification(limited_params)
    assert len(limited_result.per_query) == 2


def test_query_classification_backend_taxonomy_evaluation(tmp_path):
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2, 3, 4, 5],
            "title": ["foo"] * 5,
            "description": ["bar"] * 5,
            "category": [
                "foo / bar / baz",
                "foo / bar / bin",
                "luz / bar / bin",
                "lump / bar / bin",
                "lump / bar / booz",
            ],
        }
    )
    judgments = pd.DataFrame(
        {
            "query": ["taxonomy query"] * 5,
            "doc_id": [1, 2, 3, 4, 5],
            "grade": [2] * 5,
        }
    )
    dataset = SimpleNamespace(corpus=corpus, judgments=judgments)
    config_path = tmp_path / "query_understanding.yml"
    config_path.write_text(
        """
strategy:
  name: query_understanding_taxonomy_fixture
  type: query_understanding
  params:
    categorize:
      field: category
      enrichment_engine:
        type: dummy
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title]
""".lstrip(),
        encoding="utf-8",
    )

    def evaluate(eval_as, report_path=None):
        params = QueryClassificationParams(
            strategy_path=str(config_path),
            dataset="doug_blog",
            query="taxonomy query",
            query_threshold=0.4,
            eval_as=eval_as,
            report_path=str(report_path) if report_path is not None else None,
        )
        with patch(
            "exps.runners.query_classification.get_dataset", return_value=dataset
        ):
            return evaluate_query_classification(params).per_query.iloc[0]

    report_path = tmp_path / "taxonomy-report.pkl"
    root_row = evaluate("taxonomy[0]", report_path)
    assert root_row["expected_categories"] == ["foo", "lump"]
    assert root_row["generated_categories"] == ["foo"]
    assert root_row["recall"] == 0.5
    assert root_row["jaccard"] == 0.5

    report = pd.read_pickle(report_path)
    assert len(report) == 5
    assert report["category"].tolist() == corpus["category"].tolist()
    assert report["expected_categories"].map(bool).eq(True).all()
    assert report["generated_categories"].map(
        lambda categories: categories == ["foo"]
    ).all()
    assert report["recall"].eq(0.5).all()
    assert report["jaccard"].eq(0.5).all()
    assert report["predicted_categories"].map(
        lambda categories: categories == ["foo / bar / baz"]
    ).all()
    assert report["category_level_0"].tolist() == ["foo", "foo", "luz", "lump", "lump"]
    assert report["ground_truth_category_level_0"].map(
        lambda categories: categories == ["foo", "lump"]
    ).all()
    assert report["predicted_categories_level_0"].map(
        lambda categories: categories == ["foo"]
    ).all()
    assert report.groupby("query")["recall"].mean().mean() == root_row["recall"]
    assert report.groupby("query")["jaccard"].mean().mean() == root_row["jaccard"]

    no_report_root_row = evaluate("taxonomy[0]")
    assert no_report_root_row["expected_categories"] == root_row["expected_categories"]
    assert no_report_root_row["generated_categories"] == root_row["generated_categories"]
    assert no_report_root_row["recall"] == root_row["recall"]
    assert no_report_root_row["jaccard"] == root_row["jaccard"]

    level_one_row = evaluate("taxonomy[1]")
    assert level_one_row["expected_categories"] == ["bar"]
    assert level_one_row["generated_categories"] == ["bar"]
    assert level_one_row["recall"] == 1.0
    assert level_one_row["jaccard"] == 1.0

    direct_report_path = tmp_path / "direct-report.pkl"
    direct_row = evaluate("direct", direct_report_path)
    assert direct_row["expected_categories"] == []
    assert direct_row["generated_categories"] == ["foo / bar / baz"]
    assert direct_row["recall"] == 0.0
    assert direct_row["jaccard"] == 0.0
    direct_report = pd.read_pickle(direct_report_path)
    assert "category_level_0" not in direct_report
    assert direct_report["predicted_categories"].map(
        lambda categories: categories == ["foo / bar / baz"]
    ).all()

    multi_report_path = tmp_path / "multi-report.pkl"
    multi_params = QueryClassificationParams(
        strategy_path=str(config_path),
        dataset="doug_blog",
        query="taxonomy query",
        query_threshold=0.4,
        eval_as="taxonomy[0], taxonomy[1], direct",
        report_path=str(multi_report_path),
    )
    with patch(
        "exps.runners.query_classification.get_dataset", return_value=dataset
    ):
        multi_result = evaluate_query_classification(multi_params)

    assert multi_result.evaluations is not None
    assert list(multi_result.evaluations) == ["taxonomy[0]", "taxonomy[1]", "direct"]
    assert multi_result.evaluations["taxonomy[0]"].mean_recall == 0.5
    assert multi_result.evaluations["taxonomy[1]"].mean_recall == 1.0
    assert multi_result.evaluations["direct"].mean_recall == 0.0

    multi_report = pd.read_pickle(multi_report_path)
    assert multi_report["expected_categories_taxonomy_0"].map(
        lambda categories: categories == ["foo", "lump"]
    ).all()
    assert multi_report["expected_categories_taxonomy_1"].map(
        lambda categories: categories == ["bar"]
    ).all()
    assert multi_report["expected_categories_direct"].map(
        lambda categories: categories == []
    ).all()
    assert multi_report["recall_taxonomy_0"].eq(0.5).all()
    assert multi_report["recall_taxonomy_1"].eq(1.0).all()
    assert multi_report["recall_direct"].eq(0.0).all()
    assert multi_report["jaccard_direct"].eq(0.0).all()


def test_query_classification_backend_rejects_invalid_eval_as(tmp_path):
    config_path = _write_bm25_config(tmp_path)
    params = QueryClassificationParams(
        strategy_path=str(config_path),
        eval_as="taxonomy[nope]",
    )

    with pytest.raises(ValueError, match="eval_as"):
        evaluate_query_classification(params)


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
