from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from exps.judge.engines.common import PairwiseJudgeResult
from exps.judge.pairwise_strategy import PairwiseJudgeStrategy


def _fixture_data():
    grades = [2, 2, 1, 1, 0, 0]
    corpus = pd.DataFrame(
        {
            "doc_id": [f"d{index}" for index in range(len(grades))],
            "title": [f"title-{index}" for index in range(len(grades))],
            "description": [f"grade={grade}" for grade in grades],
        }
    )
    judgments = pd.DataFrame(
        {"query": ["desk"] * len(grades), "doc_id": corpus["doc_id"], "grade": grades}
    )
    return corpus, judgments


def _multi_query_fixture_data():
    grades = [
        1.0, 1.0, 0.5, 0.5, 0.1, 0.1, 0.0, 0.0,
        0.0, 0.0, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0,
    ]
    corpus = pd.DataFrame(
        {
            "doc_id": [f"d{index}" for index in range(len(grades))],
            "title": [f"title-{index}" for index in range(len(grades))],
            "description": [f"grade={grade}" for grade in grades],
        }
    )
    judgments = pd.DataFrame(
        {
            "query": ["desk"] * 8 + ["lamp"] * 8,
            "doc_id": corpus["doc_id"],
            "grade": grades,
        }
    )
    return corpus, judgments


def _strategy(corpus, judgments, **overrides):
    params = {
        "model": "gpt-5-mini",
        "prompt": (
            "Query: {query}\nLHS: {title_lhs} / {description_lhs}\n"
            "RHS: {title_rhs} / {description_rhs}"
        ),
        "question": "Which document is more relevant for {query}?",
        "to_pointwise": "copeland",
        "sample_size": 3,
        "sampling_seed": 42,
        "double_check": False,
    }
    params.update(overrides)
    with patch("exps.judge.pairwise_strategy.LLMPairwiseJudgeEngine") as engine:
        engine.return_value.judge.return_value = PairwiseJudgeResult(outcome="lhs")
        strategy = PairwiseJudgeStrategy(corpus, judgments=judgments, **params)
    return strategy


def test_sampling_is_deterministic_balanced_and_best_effort():
    corpus, judgments = _fixture_data()
    first = _strategy(corpus, judgments, sample_size=5)
    second = _strategy(corpus, judgments, sample_size=5)

    assert first._samples == second._samples
    assert len(first._samples["desk"]) == 6
    counts = pd.Series([grade for _, grade in first._samples["desk"]]).value_counts()
    assert max(counts) - min(counts) <= 1
    assert len(_strategy(corpus, judgments, sample_size=10)._samples["desk"]) == 6


def test_sampling_keeps_multiple_queries_separate():
    corpus, judgments = _multi_query_fixture_data()
    strategy = _strategy(corpus, judgments, sample_size=2)
    expected = {
        "desk": {
            (0, 1.0), (1, 1.0), (2, 0.5), (3, 0.5),
            (4, 0.1), (5, 0.1), (6, 0.0), (7, 0.0),
        },
        "lamp": {
            (8, 0.0), (9, 0.0), (10, 0.1), (11, 0.1),
            (12, 0.5), (13, 0.5), (14, 1.0), (15, 1.0),
        },
    }

    assert set(strategy._samples) == set(expected)
    for query, expected_samples in expected.items():
        sampled = strategy._samples[query]
        assert len(sampled) == 4
        assert set(sampled) <= expected_samples
        assert {grade for _, grade in sampled} == {0.0, 0.1, 0.5, 1.0}


def test_sampling_supports_float_grades_between_zero_and_one():
    corpus, judgments = _multi_query_fixture_data()
    strategy = _strategy(corpus, judgments, sample_size=4)
    expected_grades = {0.0, 0.1, 0.5, 1.0}

    for sampled in strategy._samples.values():
        assert len(sampled) == 4
        assert {grade for _, grade in sampled} == expected_grades


def test_pair_prompt_renders_query_and_both_document_sides():
    corpus, judgments = _fixture_data()
    strategy = _strategy(corpus, judgments)
    request = strategy._render_request("desk", 0, 4)

    assert request.prompt == (
        "Query: desk\nLHS: title-0 / grade=2\nRHS: title-4 / grade=0"
    )
    assert request.question == "Which document is more relevant for desk?"
    assert request.prompt_has_question is False


def test_prompt_question_placeholder_and_missing_field_validation():
    corpus, judgments = _fixture_data()
    strategy = _strategy(
        corpus,
        judgments,
        prompt="{query} {question}: {title_lhs} vs {title_rhs}",
    )
    request = strategy._render_request("desk", 0, 1)
    assert request.prompt == "desk Which document is more relevant for desk?: title-0 vs title-1"
    assert request.prompt_has_question is True

    with pytest.raises(ValueError, match="placeholders"):
        _strategy(corpus, judgments, prompt="{query} {missing_lhs}")
    with pytest.raises(ValueError, match="placeholders"):
        _strategy(corpus, judgments, question="{question}")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"to_pointwise": "borda"}, "to_pointwise"),
        ({"sample_size": 0}, "sample_size"),
        ({"sample_size": True}, "sample_size"),
        ({"confidence_threshold": -0.1}, "confidence_threshold"),
        ({"explicit_neither": "yes"}, "explicit_neither"),
        ({"double_check": 1}, "double_check"),
    ],
)
def test_invalid_configuration_fails(overrides, message):
    corpus, judgments = _fixture_data()
    with pytest.raises(ValueError, match=message):
        _strategy(corpus, judgments, **overrides)


def test_openai_confidence_threshold_has_actionable_error():
    corpus, judgments = _fixture_data()
    with pytest.raises(ValueError, match="requires a Jev model.*do not provide choice probabilities"):
        _strategy(corpus, judgments, confidence_threshold=0.7)


def test_pairwise_search_uses_all_pairs_and_copeland_scores():
    corpus, judgments = _fixture_data()
    strategy = _strategy(corpus, judgments, sample_size=3)
    strategy.engine.judge.side_effect = [
        PairwiseJudgeResult(outcome="rhs"),
        PairwiseJudgeResult(outcome="lhs"),
        PairwiseJudgeResult(outcome="lhs"),
    ]

    indices, scores = strategy.search("desk", k=3)

    sampled = [index for index, _ in strategy._samples["desk"]]
    assert indices == [sampled[1], sampled[0], sampled[2]]
    assert scores == [2.0, 1.0, 0.0]
    assert strategy.engine.judge.call_count == 3


def test_tournament_scheduler_limits_documents_and_runs_configured_rounds():
    corpus, judgments = _fixture_data()
    strategy = _strategy(
        corpus,
        judgments,
        sample_size=3,
        scheduler={
            "type": "tournament",
            "params": {"n_per_round": 2, "max_rounds": 2},
        },
    )

    indices, _ = strategy.search("desk", k=3)

    assert len(indices) == 3
    assert strategy.engine.judge.call_count == 2


@pytest.mark.parametrize(
    ("scheduler", "message"),
    [
        ({"type": "unknown"}, "scheduler.type"),
        ({"type": "tournament", "params": {"n_per_round": 1}}, "at least 2"),
        ({"type": "tournament", "params": {"max_rounds": 0}}, "positive integer"),
        ({"type": "tournament", "params": {"extra": 1}}, "Unknown tournament"),
    ],
)
def test_invalid_scheduler_configuration_fails(scheduler, message):
    corpus, judgments = _fixture_data()
    with pytest.raises(ValueError, match=message):
        _strategy(corpus, judgments, scheduler=scheduler)


def test_neither_scores_half_point_and_equal_scores_break_by_corpus_index():
    corpus, judgments = _fixture_data()
    strategy = _strategy(corpus, judgments, sample_size=3)
    strategy.engine.judge.return_value = PairwiseJudgeResult(outcome="neither")

    indices, scores = strategy.search("desk", k=3)

    assert indices == sorted(index for index, _ in strategy._samples["desk"])
    assert scores == [1.0, 1.0, 1.0]


def test_single_document_and_unseen_query_do_not_call_the_judge():
    corpus, judgments = _fixture_data()
    corpus = corpus.iloc[:1].reset_index(drop=True)
    judgments = judgments.iloc[:1]
    strategy = _strategy(corpus, judgments, sample_size=1)

    indices, scores = strategy.search("desk", k=10)

    assert len(indices) == 1
    assert scores == [0.0]
    assert strategy.engine.judge.call_count == 0
    assert strategy.search("unknown query", k=10) == ([], [])


def test_double_check_requires_reversed_outcome_and_swaps_rendered_fields():
    corpus, judgments = _fixture_data()
    strategy = _strategy(corpus, judgments, double_check=True)
    strategy.engine.judge.side_effect = [
        PairwiseJudgeResult(outcome="lhs"),
        PairwiseJudgeResult(outcome="rhs"),
        PairwiseJudgeResult(outcome="lhs"),
        PairwiseJudgeResult(outcome="lhs"),
    ]

    first_pair = strategy._compare("desk", 0, 1)
    inconsistent_pair = strategy._compare("desk", 2, 3)

    assert first_pair == "lhs"
    assert inconsistent_pair == "neither"
    first_prompt = strategy.engine.judge.call_args_list[0].kwargs["prompt"]
    swapped_prompt = strategy.engine.judge.call_args_list[1].kwargs["prompt"]
    assert "title-0" in first_prompt and "title-1" in first_prompt
    assert first_prompt != swapped_prompt


def test_confidence_threshold_abstains_at_or_below_threshold():
    corpus, judgments = _fixture_data()
    with patch("exps.judge.pairwise_strategy.JevPairwiseJudgeEngine") as engine:
        strategy = PairwiseJudgeStrategy(
            corpus,
            judgments=judgments,
            model="jev/jev-latest",
            prompt="{query} {title_lhs} {title_rhs}",
            question="Which?",
            confidence_threshold=0.7,
            sample_size=2,
            double_check=False,
        )
    strategy.engine = engine.return_value
    strategy.engine.judge.side_effect = [
        PairwiseJudgeResult(outcome="lhs", probability=0.7),
    ]

    assert strategy._judge_once("desk", 0, 1).outcome == "neither"
    assert strategy.engine.judge.call_args.kwargs == {
        "state": "desk title-0 title-1",
        "question": "Which?",
    }


def test_confidence_threshold_accepts_probability_above_threshold():
    corpus, judgments = _fixture_data()
    with patch("exps.judge.pairwise_strategy.JevPairwiseJudgeEngine") as engine:
        strategy = PairwiseJudgeStrategy(
            corpus,
            judgments=judgments,
            model="jev/jev-latest",
            prompt="{query} {title_lhs} {title_rhs}",
            question="Which?",
            confidence_threshold=0.7,
            sample_size=2,
            double_check=False,
        )
    strategy.engine = engine.return_value
    strategy.engine.judge.return_value = PairwiseJudgeResult(
        outcome="rhs", probability=0.71
    )

    assert strategy._judge_once("desk", 0, 1).outcome == "rhs"


def test_cache_key_includes_pairwise_behavior_configuration():
    corpus, judgments = _fixture_data()
    defaults = _strategy(corpus, judgments)
    checked = _strategy(corpus, judgments, double_check=True)
    neither = _strategy(corpus, judgments, explicit_neither=True)

    assert defaults.cache_key != checked.cache_key
    assert defaults.cache_key != neither.cache_key
