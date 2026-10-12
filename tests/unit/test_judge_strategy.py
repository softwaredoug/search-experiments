from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from exps.judge.pontwise_strategy import PointwiseJudgeStrategy


def _fixture_data():
    corpus = pd.DataFrame(
        {
            "doc_id": ["d0", "d1", "d2", "d3", "d4", "d5"],
            "title": [f"title-{index}" for index in range(6)],
            "description": [f"description-{index}" for index in range(6)],
        }
    )
    judgments = pd.DataFrame(
        {
            "query": ["desk"] * 6,
            "doc_id": corpus["doc_id"],
            "grade": [2, 2, 1, 1, 0, 0],
        }
    )
    return corpus, judgments


def _strategy(corpus, judgments, **overrides):
    params = {
        "model": "gpt-5-mini",
        "prompt": "Query: {query}\nTitle: {title}\nDescription: {description}",
        "question": "Select the best rubric grade for {query}.",
        "rubric": {2: "relevant", 1: "partially relevant", 0: "irrelevant"},
        "sample_size": 3,
        "sampling_seed": 42,
    }
    params.update(overrides)
    with patch("exps.judge.pontwise_strategy.LLMJudgeEngine") as engine:
        instance = engine.return_value
        instance.judge.return_value = SimpleNamespace(grade=1, probability=None)
        strategy = PointwiseJudgeStrategy(
            corpus, judgments=judgments, **params
        )
    return strategy


def test_sampling_is_seeded_balanced_and_limited_to_available_documents():
    corpus, judgments = _fixture_data()
    first = _strategy(corpus, judgments, sample_size=5)
    second = _strategy(corpus, judgments, sample_size=5)

    assert first._samples == second._samples
    assert len(first._samples["desk"]) == 5
    counts = pd.Series([grade for _, grade in first._samples["desk"]]).value_counts()
    assert max(counts) - min(counts) <= 1


def test_prompt_and_question_are_rendered_from_query_and_corpus_fields():
    corpus, judgments = _fixture_data()
    strategy = _strategy(
        corpus,
        judgments,
        prompt="{query}: {title} — {description}",
        question="Grade this result for {query}: {title}",
    )

    request = strategy._render_request("desk", 0)
    assert request.prompt == "desk: title-0 — description-0"
    assert request.question == "Grade this result for desk: title-0"
    assert request.prompt_has_question is False
    assert request.prompt_has_rubric is False


def test_template_placeholder_errors_are_reported():
    corpus, judgments = _fixture_data()
    with pytest.raises(ValueError, match="placeholders"):
        _strategy(corpus, judgments, prompt="{missing_column}")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"question": "  "}, "question"),
        ({"rubric": {2: "relevant", 1: "partial"}}, "missing judgment grade"),
        ({"sample_size": 0}, "sample_size"),
    ],
)
def test_invalid_judge_configuration_fails(overrides, message):
    corpus, judgments = _fixture_data()
    with pytest.raises(ValueError, match=message):
        _strategy(corpus, judgments, **overrides)


def test_cache_key_includes_explicit_question():
    corpus, judgments = _fixture_data()
    first = _strategy(corpus, judgments, question="Question A")
    second = _strategy(corpus, judgments, question="Question B")

    assert first.cache_key != second.cache_key


def test_search_sorts_sampled_documents_by_predicted_grade():
    corpus, judgments = _fixture_data()
    strategy = _strategy(corpus, judgments, sample_size=3)
    grades = iter([0, 2, 1])
    strategy._judge = lambda _query, _index, **_kwargs: next(grades)

    indices, scores = strategy.search("desk", k=3)

    assert scores == [2, 1, 0]
    assert len(indices) == 3
