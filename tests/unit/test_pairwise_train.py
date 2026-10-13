from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from exps.judge.pairwise_train import (
    PairwiseFeatureComputer,
    _normalize_model_config,
    _new_pipeline,
    _pairwise_prediction_metrics,
    _predict_pairwise_outcomes,
    build_binary_training_data,
    query_group_folds,
)


def test_binary_training_data_drops_grade_ties_and_targets_the_higher_grade_side():
    comparisons = pd.DataFrame(
        [
            {
                "query": "query-a",
                "grade_lhs": 2,
                "grade_rhs": 0,
                "title": "lhs",
            },
            {
                "query": "query-b",
                "grade_lhs": 0,
                "grade_rhs": 1,
                "title": "rhs",
            },
            {
                "query": "query-c",
                "grade_lhs": 1,
                "grade_rhs": 1,
                "title": "neither",
            },
        ]
    )

    features, targets, groups = build_binary_training_data(
        comparisons,
        feature_columns=["title"],
    )

    assert features.to_dict("records") == [{"title": "lhs"}, {"title": "rhs"}]
    assert targets.tolist() == ["lhs", "rhs"]
    assert groups.tolist() == ["query-a", "query-b"]


def test_neither_feature_output_is_retained_as_a_classifier_input():
    comparisons = pd.DataFrame(
        [
            {"query": "query-a", "grade_lhs": 2, "grade_rhs": 1, "title": "neither"},
            {"query": "query-b", "grade_lhs": 0, "grade_rhs": 1, "title": "rhs"},
        ]
    )
    features, targets, _ = build_binary_training_data(
        comparisons,
        feature_columns=["title"],
    )
    model = _new_pipeline("decision_tree")
    model.fit(features, targets)

    assert features["title"].tolist() == ["neither", "rhs"]
    assert model.named_steps["onehotencoder"].categories_[0].tolist() == [
        "neither",
        "rhs",
    ]


def test_bare_corpus_placeholder_error_explains_pairwise_side_suffixes():
    corpus = pd.DataFrame({"doc_id": [1], "category": ["chairs"]})
    features = [
        {
            "category": {
                "model": "jev/jev-latest",
                "prompt": "{query} {category}",
                "question": "Which is more relevant?",
            }
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        PairwiseFeatureComputer(corpus=corpus, features=features)

    message = str(exc_info.value)
    assert "unsupported prompt placeholder(s): category" in message
    assert "corpus columns must be suffixed with '_lhs' or '_rhs'" in message
    assert "{category_lhs} and {category_rhs}" in message


@pytest.mark.parametrize("empty_value", [None, float("nan"), pd.NA, "", "   ", [], {}])
def test_pairwise_feature_uses_configured_empty_value_for_empty_prompt_inputs(empty_value):
    corpus = pd.DataFrame({"doc_id": [1], "category": ["chairs"]})
    features = [
        {
            "category": {
                "model": "gpt-5-mini",
                "empty_value": "No Description Provided",
                "prompt": "LHS: {category_lhs}\nRHS: {category_rhs}",
                "question": "Which is more relevant?",
            }
        }
    ]

    class CapturingEngine:
        def judge(self, *, prompt, **_kwargs):
            self.prompt = prompt
            return SimpleNamespace(outcome="lhs")

    feature_computer = PairwiseFeatureComputer(corpus=corpus, features=features)
    engine = CapturingEngine()
    feature_computer._judge(
        feature_computer.features[0],
        engine,
        "query",
        {"category": empty_value},
        {"category": empty_value},
    )

    assert engine.prompt == (
        "LHS: No Description Provided\nRHS: No Description Provided"
    )


def test_pairwise_feature_defaults_empty_prompt_values_to_unknown():
    corpus = pd.DataFrame({"doc_id": [1], "category": ["chairs"]})
    features = [
        {
            "category": {
                "model": "gpt-5-mini",
                "prompt": "LHS: {category_lhs}\nRHS: {category_rhs}",
                "question": "Which is more relevant?",
            }
        }
    ]

    class CapturingEngine:
        def judge(self, *, prompt, **_kwargs):
            self.prompt = prompt
            return SimpleNamespace(outcome="lhs")

    feature_computer = PairwiseFeatureComputer(corpus=corpus, features=features)
    engine = CapturingEngine()
    feature_computer._judge(
        feature_computer.features[0],
        engine,
        "query",
        {"category": None},
        {"category": "  "},
    )

    assert engine.prompt == "LHS: Unknown\nRHS: Unknown"


@pytest.mark.parametrize("empty_value", ["", "   ", None, 42])
def test_pairwise_feature_rejects_invalid_empty_value(empty_value):
    corpus = pd.DataFrame({"doc_id": [1], "category": ["chairs"]})
    features = [
        {
            "category": {
                "model": "gpt-5-mini",
                "empty_value": empty_value,
                "prompt": "{category_lhs} {category_rhs}",
                "question": "Which is more relevant?",
            }
        }
    ]

    with pytest.raises(ValueError, match="empty_value must be a non-empty string"):
        PairwiseFeatureComputer(corpus=corpus, features=features)


def test_query_group_folds_keep_each_query_entirely_in_one_validation_fold():
    groups = ["q1", "q1", "q2", "q2", "q3", "q3", "q4", "q4"]

    folds = query_group_folds(groups, n_splits=4)

    assert len(folds) == 4
    validation_queries = []
    for train_indices, validation_indices in folds:
        train_queries = {groups[index] for index in train_indices}
        fold_validation_queries = {groups[index] for index in validation_indices}
        assert train_queries.isdisjoint(fold_validation_queries)
        validation_queries.extend(fold_validation_queries)
    assert sorted(validation_queries) == ["q1", "q2", "q3", "q4"]


@pytest.mark.parametrize(
    ("groups", "n_splits", "message"),
    [
        (["q1", "q2"], 1, "at least 2"),
        (["q1", "q2"], 3, "distinct training queries"),
    ],
)
def test_query_group_folds_rejects_invalid_fold_counts(groups, n_splits, message):
    with pytest.raises(ValueError, match=message):
        query_group_folds(groups, n_splits=n_splits)


def test_model_config_normalizes_prediction_threshold():
    assert _normalize_model_config(
        {"type": "decision_tree", "params": {"prediction_threshold": 0.7}}
    ) == {
        "type": "decision_tree",
        "params": {"prediction_threshold": 0.7},
    }


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True])
def test_model_config_rejects_invalid_prediction_threshold(threshold):
    with pytest.raises(ValueError, match="prediction_threshold"):
        _normalize_model_config(
            {"type": "decision_tree", "params": {"prediction_threshold": threshold}}
        )


def test_prediction_threshold_abstains_below_confidence_threshold():
    class FakeModel:
        def predict(self, _features):
            return ["lhs", "lhs", "rhs"]

        def predict_proba(self, _features):
            return [[0.8, 0.2], [0.55, 0.45], [0.3, 0.7]]

    assert _predict_pairwise_outcomes(FakeModel(), None, 0.7) == [
        "lhs",
        "neither",
        "rhs",
    ]
    assert _predict_pairwise_outcomes(FakeModel(), None, None) == ["lhs", "lhs", "rhs"]


def test_pairwise_metrics_report_accuracy_and_precision_for_decided_pairs():
    metrics = _pairwise_prediction_metrics(
        ["lhs", "rhs", "lhs", "rhs"],
        ["lhs", "neither", "rhs", "rhs"],
    )

    assert metrics["accuracy"] == 0.5
    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["prediction_coverage"] == 0.75
    assert metrics["decided_pair_count"] == 3
    assert metrics["correct_decided_pair_count"] == 2


def test_pairwise_precision_is_undefined_when_every_prediction_abstains():
    metrics = _pairwise_prediction_metrics(["lhs", "rhs"], ["neither", "neither"])

    assert metrics["accuracy"] == 0.0
    assert metrics["precision"] is None
    assert metrics["prediction_coverage"] == 0.0


def test_pairwise_trained_judge_gives_half_a_point_to_each_abstained_pair():
    from exps.judge.pairwise_trained_strategy import PairwiseTrainedJudgeStrategy

    class FakeFeatureComputer:
        def compute(self, *_args):
            return {"signal": "ambiguous"}

    class FakeModel:
        def predict(self, _features):
            return ["lhs"]

        def predict_proba(self, _features):
            return [[0.6, 0.4]]

    strategy = PairwiseTrainedJudgeStrategy.__new__(PairwiseTrainedJudgeStrategy)
    strategy.samples = {"query": [(0, 1), (1, 0)]}
    strategy.feature_computer = FakeFeatureComputer()
    strategy.corpus = pd.DataFrame({"doc_id": [0, 1]})
    strategy.feature_names = ["signal"]
    strategy.model = FakeModel()
    strategy.prediction_threshold = 0.7

    indices, scores = strategy.search("query")

    assert indices == [0, 1]
    assert scores == [0.5, 0.5]


def test_pairwise_feature_computer_shares_jev_cache_and_clears_it_once():
    class FakeAutoDecisionEnricher:
        instances = []

        def __init__(self, *, choices, model):
            self.choices = choices
            self.model = model
            self.delete_cache_calls = 0
            self.save_cache_calls = 0
            self.cached_enricher = self
            type(self).instances.append(self)

        def delete_cache(self):
            self.delete_cache_calls += 1

        def save_cache(self):
            self.save_cache_calls += 1

    corpus = pd.DataFrame({"doc_id": [1], "title": ["desk lamp"]})
    features = [
        {
            "title": {
                "model": "jev/jev-latest",
                "prompt": "{query} {title_lhs} {title_rhs}",
                "question": "Which is more relevant?",
            }
        },
        {
            "title_detail": {
                "model": "jev/jev-latest",
                "prompt": "{query} {title_lhs} {title_rhs}",
                "question": "Which is more relevant?",
            }
        },
    ]
    FakeAutoDecisionEnricher.instances = []

    with patch(
        "exps.judge.engines.jev_pairwise.AutoDecisionEnricher",
        FakeAutoDecisionEnricher,
    ):
        feature_computer = PairwiseFeatureComputer(
            corpus=corpus,
            features=features,
            no_cache=True,
        )
        feature_computer.flush_caches()

    assert feature_computer.engines[0] is feature_computer.engines[1]
    assert len(FakeAutoDecisionEnricher.instances) == 1
    assert FakeAutoDecisionEnricher.instances[0].delete_cache_calls == 1
    assert FakeAutoDecisionEnricher.instances[0].save_cache_calls == 1
