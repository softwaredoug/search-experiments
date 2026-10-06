"""End-to-end tests for query understanding and category classification."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from exps.query_classification_runner import main as query_classification_main
from exps.runners.query_classification import (
    QueryClassificationParams,
    evaluate_query_classification,
)
from exps.runners.run import RunParams, run_benchmark


class ScriptedJevClient:
    """A deterministic stand-in for Jev at the SDK boundary."""

    predictions = {}
    instances = []

    def __init__(self, **kwargs):
        self.api_key = kwargs["api_key"]
        self.model = kwargs["model"]
        self.calls = []
        self.__class__.instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        question_id = next(iter(questions))
        answer = type(
            "Answer",
            (),
            {"choice": self.predictions[state], "confidence": 1.0},
        )()
        return type("Response", (), {"choices": {question_id: answer}})()


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
    with patch("exps.runners.query_classification.get_dataset", return_value=dataset):
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
    with patch("exps.runners.query_classification.get_dataset", return_value=dataset):
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
    with patch("exps.runners.query_classification.get_dataset", return_value=dataset):
        limited_result = evaluate_query_classification(limited_params)
    assert len(limited_result.per_query) == 2


def test_query_classification_jev_empty_choices_e2e(fake_wands_dataset, tmp_path):
    ScriptedJevClient.instances = []
    ScriptedJevClient.predictions = {"floating bed": "Furniture"}
    config_path = tmp_path / "query_understanding_jev.yml"
    config_path.write_text(
        """
strategy:
  name: query_understanding_jev_empty_choices_fixture
  type: query_understanding
  params:
    categorize:
      field: category
      enrichment_engine:
        type: choice_single
        params:
          model: jev/jev-latest
          prompt: Classify {query} into a product category.
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title, description]
        boost_matches: 10
""".lstrip(),
        encoding="utf-8",
    )

    with (
        patch(
            "exps.runners.query_classification.get_dataset",
            return_value=fake_wands_dataset,
        ),
        patch(
            "exps.query_understanding.enrichers.choice_single_jev.TypeSafeClient",
            ScriptedJevClient,
        ),
        patch(
            "exps.query_understanding.enrichers.choice_single_jev.key_for_provider",
            return_value="typesafe-test-key",
        ),
        patch(
            "exps.query_understanding.enrichers.cached_choice_single_jev.DATA_PATH",
            tmp_path,
        ),
    ):
        result = evaluate_query_classification(
            QueryClassificationParams(
                strategy_path=str(config_path),
                dataset="wands",
                query="floating bed",
            )
        )

    row = result.per_query.iloc[0]
    assert row["expected_categories"] == ["Furniture"]
    assert row["generated_categories"] == ["Furniture"]
    assert row["recall"] == 1.0
    assert row["jaccard"] == 1.0
    assert result.coverage == 1.0

    _, questions = ScriptedJevClient.instances[0].calls[0]
    assert questions["category"].criteria == {
        "Furniture": None,
        "Bedroom": None,
        "Unknown": "No classification applies.",
    }


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
    assert report["has_prediction"].all()
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
    assert report["category_level_0"].tolist() == [
        "foo",
        "foo",
        "luz",
        "lump",
        "lump",
    ]
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
    assert multi_report["has_prediction"].all()
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


def test_query_classification_report_mean_matches_csv_and_cli(
    tmp_path, capsys
):
    judgment_queries = [
        "correct",
        "wrong",
        "wrong",
        "no_ground_truth",
        "no_ground_truth",
        "no_ground_truth",
        "abstain",
        "abstain",
        "abstain",
        "abstain",
    ]
    categories = [
        "Furniture",
        "Lighting",
        "Lighting",
        "Outdoor",
        "Rugs",
        "Bed & Bath",
        "Garden",
        "Garden",
        "Garden",
        "Garden",
    ]
    doc_ids = list(range(1, len(judgment_queries) + 1))
    corpus = pd.DataFrame(
        {
            "doc_id": doc_ids,
            "category": categories,
            "title": ["sample product"] * len(doc_ids),
        }
    )
    judgments = pd.DataFrame(
        {"query": judgment_queries, "doc_id": doc_ids, "grade": [2] * len(doc_ids)}
    )
    dataset = SimpleNamespace(corpus=corpus, judgments=judgments)

    config_path = tmp_path / "query_understanding.yml"
    config_path.write_text(
        """
strategy:
  name: query_classification_report_fixture
  type: query_understanding
  params:
    categorize:
      field: category
      enrichment_engine:
        type: choice_single
        params:
          model: jev/jev-latest
          choices: {}
          prompt: Classify {query}.
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title]
""".lstrip(),
        encoding="utf-8",
    )

    ScriptedJevClient.instances = []
    ScriptedJevClient.predictions = {
        "correct": "Furniture",
        "wrong": "Furniture",
        "no_ground_truth": "Outdoor",
        "abstain": "Unknown",
    }
    report_path = tmp_path / "class-report.pkl"
    summary_path = tmp_path / "summary.csv"
    with (
        patch("exps.runners.query_classification.get_dataset", return_value=dataset),
        patch(
            "exps.query_understanding.enrichers.choice_single_jev.TypeSafeClient",
            ScriptedJevClient,
        ),
        patch(
            "exps.query_understanding.enrichers.choice_single_jev.key_for_provider",
            return_value="typesafe-test-key",
        ),
        patch(
            "exps.query_understanding.enrichers.cached_choice_single_jev.DATA_PATH",
            tmp_path,
        ),
        patch(
            "sys.argv",
            [
                "query_classification",
                "--strategy",
                str(config_path),
                "--dataset",
                "doug_blog",
                "--eval-as",
                "direct",
                "--query-threshold",
                "0.8",
                "--report",
                str(report_path),
                "--summary-csv",
                str(summary_path),
            ],
        ),
    ):
        query_classification_main()

    report = pd.read_pickle(report_path)
    assert report.loc[report["query"] == "no_ground_truth", "recall"].eq(0.0).all()
    assert not report.loc[report["query"] == "abstain", "has_prediction"].any()
    report_mean_recall = (
        report.loc[report["has_prediction"]]
        .groupby("query")["recall"]
        .mean()
        .mean()
    )

    summary = pd.read_csv(summary_path)
    assert summary.loc[0, "mean_recall"] == pytest.approx(report_mean_recall)
    assert report_mean_recall == pytest.approx(1 / 3)
    assert f"Mean recall: {report_mean_recall:.4f}" in capsys.readouterr().out


def test_query_classification_backend_rejects_invalid_eval_as(tmp_path):
    config_path = tmp_path / "query_understanding.yml"
    config_path.write_text(
        """
strategy:
  name: query_classification_invalid_eval_fixture
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
    params = QueryClassificationParams(
        strategy_path=str(config_path),
        eval_as="taxonomy[nope]",
    )

    with pytest.raises(ValueError, match="eval_as"):
        evaluate_query_classification(params)
