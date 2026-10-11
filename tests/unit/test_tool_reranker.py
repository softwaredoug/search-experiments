from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from exps.tools.reranker import make_reranker


class _ScriptedJevClient:
    calls = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        type(self).calls = []

    def system_one(self, *, state, questions):
        question = questions["relevance"]
        self.calls.append((state, question))
        probability, confidence = {
            "Query: desk\nAlpha\nDescription A": (0.1, 0.95),
            "Query: desk\nBeta\nDescription B": (0.9, 0.9),
            "Query: desk\nGamma\nDescription C": (0.99, 0.4),
        }[state]
        return SimpleNamespace(
            answers={
                "relevance": SimpleNamespace(
                    probabilities={"Relevant": probability, "Not Relevant": 1 - probability},
                    confidence=confidence,
                )
            }
        )


class _ScriptedQuestionGenerator:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.response_model = kwargs["response_model"]
        self.cached_calls = []
        self.uncached_calls = []
        self.enricher = SimpleNamespace(enrich=self.enrich_uncached)
        type(self).instances.append(self)

    def enrich_uncached(self, prompt):
        self.uncached_calls.append(prompt)
        return self.response_model(
            decisions=["Is it a desk?", "Is it useful for studying?", "Is it a desk?"]
        )

    def enrich(self, prompt):
        self.cached_calls.append(prompt)
        return self.response_model(
            decisions=["Is it a desk?", "Is it useful for studying?", "Is it a desk?"]
        )


class _ScriptedNoulClient:
    instances = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        self.calls = []
        type(self).instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        title = state.splitlines()[1]
        probabilities = {
            "Alpha": {"decision_0": 0.1, "decision_1": 0.2},
            "Beta": {"decision_0": 0.9, "decision_1": 0.8},
        }[title]
        return SimpleNamespace(
            answers={
                question_id: SimpleNamespace(noul=probabilities[question_id])
                for question_id in questions
            }
        )


class _DecisionAnnotationNoulClient:
    instances = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        type(self).instances.append(self)

    def system_one(self, *, state, questions):
        title = state.splitlines()[1]
        probabilities = {
            "Alpha": {"decision_0": 0.7, "decision_1": 0.3},
            "Beta": {"decision_0": 0.3, "decision_1": 0.7},
            "Gamma": {"decision_0": 0.5, "decision_1": None},
        }[title]
        return SimpleNamespace(
            answers={
                question_id: SimpleNamespace(noul=probabilities[question_id])
                for question_id in questions
                if probabilities[question_id] is not None
            }
        )


def _corpus():
    return pd.DataFrame(
        {
            "doc_id": [1, 2, 3, 4],
            "title": ["Alpha", "Beta", "Gamma", "Delta"],
            "description": [
                "Description A",
                "Description B",
                "Description C",
                "Description D",
            ],
        }
    )


def _reranker_config(**updates):
    params = {
        "decision_model": "jev/jev-latest",
        "decision_weight": 10,
        "confidence_threshold": 0.7,
        "state_format": "Query: {query}\n{title}\n{description}",
        "prompt": "Is this document relevant to {query}?",
    }
    params.update(updates)
    return {"type": "jev", "k": 3, "params": params}


def test_jev_reranker_adds_weighted_probability_and_sorts_candidates():
    with (
        patch("exps.tools.reranker.TypeSafeClient", _ScriptedJevClient),
        patch("exps.tools.reranker.key_for_provider", return_value="typesafe-test-key"),
    ):
        reranker = make_reranker(_corpus(), _reranker_config())
        results = reranker.rerank(
            query="desk",
            candidates=[
                {"id": 1, "title": "Alpha", "description": "Description A", "score": 3.0},
                {"id": 2, "title": "Beta", "description": "Description B", "score": 1.0},
                {"id": 3, "title": "Gamma", "description": "Description C", "score": 2.0},
                {"id": 4, "title": "Delta", "description": "Description D", "score": 0.5},
            ],
        )

    assert [candidate["id"] for candidate in results] == [2, 1, 3, 4]
    assert [candidate["score"] for candidate in results] == pytest.approx(
        [10.0, 4.0, 2.0, 0.5]
    )
    assert len(_ScriptedJevClient.calls) == 3
    state, question = _ScriptedJevClient.calls[0]
    assert state == "Query: desk\nAlpha\nDescription A"
    assert question.instructions == "Is this document relevant to desk?"
    assert question.criteria["Relevant"]


def test_bag_of_decisions_generator_sums_nouls_and_reranks_candidates():
    _ScriptedQuestionGenerator.instances = []
    _ScriptedNoulClient.instances = []
    config = {
        "type": "bag_of_decisions",
        "k": 2,
        "params": {
            "decision_model": "jev/jev-latest",
            "decision_weight": 10,
            "state_format": "Query: {query}\n{title}\n{description}",
            "generator": {
                "type": "llm",
                "model": "gpt-5-mini",
                "system_prompt": "Generate relevance questions.",
                "prompt": "Generate yes/no questions for {query}.",
                "no_cache": True,
            },
        },
    }

    with (
        patch("exps.tools.reranker.AutoEnricher", _ScriptedQuestionGenerator),
        patch("exps.tools.reranker.TypeSafeClient", _ScriptedNoulClient),
        patch("exps.tools.reranker.key_for_provider", return_value="typesafe-test-key"),
    ):
        reranker = make_reranker(_corpus(), config)
        results = reranker.rerank(
            query="desk",
            candidates=[
                {"id": 1, "title": "Alpha", "description": "Description A", "score": 1.0},
                {"id": 2, "title": "Beta", "description": "Description B", "score": 0.5},
                {"id": 3, "title": "Gamma", "description": "Description C", "score": 0.1},
            ],
        )

    assert [candidate["id"] for candidate in results] == [2, 1, 3]
    assert [candidate["score"] for candidate in results] == pytest.approx(
        [17.5, 4.0, 0.1]
    )
    assert all(
        not {
            "requirements_met",
            "requirements_failed",
            "requirements_uncertain",
        }.intersection(candidate)
        for candidate in results
    )
    generator = _ScriptedQuestionGenerator.instances[0]
    assert generator.uncached_calls == ["Generate yes/no questions for desk."]
    assert generator.cached_calls == []
    client = _ScriptedNoulClient.instances[0]
    assert len(client.calls) == 2
    assert [question.instructions for question in client.calls[0][1].values()] == [
        "Is it a desk?",
        "Is it useful for studying?",
    ]
    assert _ScriptedNoulClient.instances[0].model == "jev-latest"


def test_bag_of_decisions_can_expose_decision_annotations_without_changing_scores():
    _ScriptedQuestionGenerator.instances = []
    _DecisionAnnotationNoulClient.instances = []
    config = {
        "type": "bag_of_decisions",
        "k": 3,
        "params": {
            "decision_model": "jev/jev-latest",
            "decision_weight": 10,
            "state_format": "Query: {query}\n{title}\n{description}",
            "expose_decisions": True,
            "generator": {
                "type": "llm",
                "model": "gpt-5-mini",
                "system_prompt": "Generate relevance questions.",
                "prompt": "Generate yes/no questions for {query}.",
            },
        },
    }

    with (
        patch("exps.tools.reranker.AutoEnricher", _ScriptedQuestionGenerator),
        patch("exps.tools.reranker.TypeSafeClient", _DecisionAnnotationNoulClient),
        patch("exps.tools.reranker.key_for_provider", return_value="typesafe-test-key"),
    ):
        reranker = make_reranker(_corpus(), config)
        results = reranker.rerank(
            query="desk",
            candidates=[
                {"id": 1, "title": "Alpha", "description": "Description A", "score": 1.0},
                {"id": 2, "title": "Beta", "description": "Description B", "score": 0.5},
                {"id": 3, "title": "Gamma", "description": "Description C", "score": 0.1},
                {"id": 4, "title": "Delta", "description": "Description D", "score": 0.0},
            ],
        )

    by_title = {candidate["title"]: candidate for candidate in results}
    assert by_title["Alpha"]["requirements_met"] == ["✅ Is it a desk?"]
    assert by_title["Alpha"]["requirements_failed"] == [
        "❌ Is it useful for studying?"
    ]
    assert by_title["Alpha"]["requirements_uncertain"] == []
    assert by_title["Beta"]["requirements_met"] == [
        "✅ Is it useful for studying?"
    ]
    assert by_title["Beta"]["requirements_failed"] == ["❌ Is it a desk?"]
    assert by_title["Gamma"]["requirements_met"] == []
    assert by_title["Gamma"]["requirements_failed"] == []
    assert by_title["Gamma"]["requirements_uncertain"] == [
        "🤷 Is it a desk?",
        "🤷 Is it useful for studying?",
    ]
    assert not {
        "requirements_met",
        "requirements_failed",
        "requirements_uncertain",
    }.intersection(by_title["Delta"])
    assert [candidate["score"] for candidate in results] == pytest.approx(
        [11.0, 10.5, 5.1, 0.0]
    )


def test_make_reranker_returns_none_when_not_configured():
    assert make_reranker(_corpus(), None) is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"type": "llm"}, "supports types 'jev' and 'bag_of_decisions'"),
        (_reranker_config(decision_model="gpt-5"), "must be a Jev model"),
        (_reranker_config(confidence_threshold=1.1), "confidence_threshold"),
        (_reranker_config(state_format="{missing}"), "missing corpus fields"),
    ],
)
def test_make_reranker_validates_configuration(config, message):
    with (
        patch("exps.tools.reranker.TypeSafeClient", _ScriptedJevClient),
        patch("exps.tools.reranker.key_for_provider", return_value="typesafe-test-key"),
        pytest.raises(ValueError, match=message),
    ):
        make_reranker(_corpus(), config)
