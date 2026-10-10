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


def test_make_reranker_returns_none_when_not_configured():
    assert make_reranker(_corpus(), None) is None


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"type": "llm"}, "supports only type 'jev'"),
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
