from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from exps.judge.engines.jev_pairwise import JevPairwiseJudgeEngine
from exps.judge.engines.llm_pairwise import LLMPairwiseJudgeEngine


class _FakeAutoEnricher:
    instances = []

    def __init__(self, *, model, system_prompt, response_model):
        self.model = model
        self.system_prompt = system_prompt
        self.response_model = response_model
        self.cached_calls = []
        self.uncached_calls = []
        self.enricher = SimpleNamespace(enrich=self._uncached_enrich)
        self.response = None
        type(self).instances.append(self)

    def enrich(self, prompt):
        self.cached_calls.append(prompt)
        return self.response

    def _uncached_enrich(self, prompt):
        self.uncached_calls.append(prompt)
        return self.response


class _FakeAutoDecisionEnricher:
    instances = []

    def __init__(self, *, choices, model):
        self.choices = choices
        self.model = model
        self.score_calls = []
        self.probabilities = {"lhs": 0.91, "rhs": 0.09}
        self.delete_cache_calls = 0
        type(self).instances.append(self)

    def score(self, *, state, question):
        self.score_calls.append((state, question))
        return self.probabilities, 0.95

    def delete_cache(self):
        self.delete_cache_calls += 1


def test_llm_engine_formats_outcome_and_normalizes_model():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pairwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMPairwiseJudgeEngine(
            model="gpt-5-mini", system_prompt="Compare carefully."
        )
        fake = _FakeAutoEnricher.instances[0]
        fake.response = fake.response_model(winner="rhs")
        result = engine.judge(
            prompt="Query: desk\nLHS: left\nRHS: right",
            question="Which is more relevant?",
        )

    assert engine.model == "openai/gpt-5-mini"
    assert result.outcome == "rhs"
    assert fake.cached_calls == [
        "Query: desk\nLHS: left\nRHS: right\n\nWhich is more relevant?"
    ]


def test_llm_engine_does_not_duplicate_question_placeholder():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pairwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMPairwiseJudgeEngine(model="openai/gpt-5-mini", system_prompt="Judge.")
        fake = _FakeAutoEnricher.instances[0]
        fake.response = fake.response_model(winner="lhs")
        engine.judge(
            prompt="Question: Which?\nLHS: a\nRHS: b",
            question="Which?",
            prompt_has_question=True,
        )

    assert fake.cached_calls == ["Question: Which?\nLHS: a\nRHS: b"]


def test_llm_engine_explicit_neither_and_no_cache():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pairwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMPairwiseJudgeEngine(
            model="gpt-5-mini",
            system_prompt="Judge.",
            explicit_neither=True,
            no_cache=True,
        )
        fake = _FakeAutoEnricher.instances[0]
        fake.response = fake.response_model(winner="neither")
        result = engine.judge(prompt="state", question="question")

    assert result.outcome == "neither"
    assert fake.cached_calls == []
    assert fake.uncached_calls == ["state\n\nquestion"]


def test_llm_engine_rejects_neither_when_not_configured():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pairwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMPairwiseJudgeEngine(model="gpt-5-mini", system_prompt="Judge.")
        fake = _FakeAutoEnricher.instances[0]
        fake.response = SimpleNamespace(winner="neither")
        with pytest.raises(ValueError, match="invalid pairwise outcome"):
            engine.judge(prompt="state", question="question")


@pytest.mark.parametrize("field", ["prompt", "question"])
def test_llm_engine_requires_non_empty_prompt_and_question(field):
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pairwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMPairwiseJudgeEngine(model="gpt-5-mini", system_prompt="Judge.")
        with pytest.raises(ValueError, match="non-empty"):
            engine.judge(
                prompt="" if field == "prompt" else "state",
                question="" if field == "question" else "question",
            )


def test_jev_engine_sends_choice_and_returns_probability():
    _FakeAutoDecisionEnricher.instances = []
    with patch(
        "exps.judge.engines.jev_pairwise.AutoDecisionEnricher",
        _FakeAutoDecisionEnricher,
    ):
        engine = JevPairwiseJudgeEngine(model="jev/jev-latest", explicit_neither=True)
        result = engine.judge(
            state="Query: desk\nLHS: left\nRHS: right",
            question="Which is more relevant?",
        )

    client = _FakeAutoDecisionEnricher.instances[0]
    assert client.model == "jev/jev-latest"
    assert result.outcome == "lhs"
    assert result.probability == 0.91
    assert client.score_calls == [
        ("Query: desk\nLHS: left\nRHS: right", "Which is more relevant?")
    ]
    assert client.choices == {
        "lhs": "The left-hand document is more relevant to the query.",
        "rhs": "The right-hand document is more relevant to the query.",
        "neither": "Neither document is more relevant than the other.",
    }


def test_jev_engine_omits_neither_criterion_by_default():
    _FakeAutoDecisionEnricher.instances = []
    with patch(
        "exps.judge.engines.jev_pairwise.AutoDecisionEnricher",
        _FakeAutoDecisionEnricher,
    ):
        engine = JevPairwiseJudgeEngine(model="jev/jev-latest")
        engine.judge(state="state", question="question")
    assert engine.criteria == {
        "lhs": "The left-hand document is more relevant to the query.",
        "rhs": "The right-hand document is more relevant to the query.",
    }


@pytest.mark.parametrize("choice", ["unknown", None])
def test_jev_engine_rejects_unknown_or_missing_choice(choice):
    _FakeAutoDecisionEnricher.instances = []
    with patch(
        "exps.judge.engines.jev_pairwise.AutoDecisionEnricher",
        _FakeAutoDecisionEnricher,
    ):
        engine = JevPairwiseJudgeEngine(model="jev/jev-latest")
        _FakeAutoDecisionEnricher.instances[0].probabilities = {choice: 0.9}
        with pytest.raises(ValueError, match="invalid pairwise outcome"):
            engine.judge(state="state", question="question")


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan"), True, "0.9"])
def test_jev_engine_rejects_invalid_probability(probability):
    _FakeAutoDecisionEnricher.instances = []
    with patch(
        "exps.judge.engines.jev_pairwise.AutoDecisionEnricher",
        _FakeAutoDecisionEnricher,
    ):
        engine = JevPairwiseJudgeEngine(model="jev/jev-latest")
        _FakeAutoDecisionEnricher.instances[0].probabilities = {
            "lhs": probability,
            "rhs": 1 - probability if isinstance(probability, (int, float)) else 0,
        }
        with pytest.raises(ValueError, match="invalid probability"):
            engine.judge(state="state", question="question")


def test_jev_engine_deletes_cache_when_no_cache_is_requested():
    _FakeAutoDecisionEnricher.instances = []
    with patch(
        "exps.judge.engines.jev_pairwise.AutoDecisionEnricher",
        _FakeAutoDecisionEnricher,
    ):
        JevPairwiseJudgeEngine(model="jev/jev-latest", no_cache=True)

    assert _FakeAutoDecisionEnricher.instances[0].delete_cache_calls == 1
