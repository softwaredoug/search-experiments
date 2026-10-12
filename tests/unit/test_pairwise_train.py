import pandas as pd
import pytest
from unittest.mock import patch

from exps.judge.pairwise_train import (
    PairwiseFeatureComputer,
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
