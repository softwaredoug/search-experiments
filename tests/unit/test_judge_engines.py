from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from exps.judge.engines.jev_pointwise import JevJudgeEngine
from exps.judge.engines.llm_pointwise import LLMJudgeEngine


RUBRIC = {2: "relevant", 1: "partially relevant", 0: "irrelevant"}


class _FakeAutoEnricher:
    instances = []

    def __init__(self, *, model, system_prompt, response_model):
        self.model = model
        self.system_prompt = system_prompt
        self.response_model = response_model
        self.prompts = []
        self.cached_calls = []
        self.uncached_calls = []
        self.enricher = SimpleNamespace(enrich=self._uncached_enrich)
        self.responses = []
        type(self).instances.append(self)

    def enrich(self, prompt):
        self.cached_calls.append(prompt)
        return self._respond(prompt)

    def _uncached_enrich(self, prompt):
        self.uncached_calls.append(prompt)
        return self._respond(prompt)

    def _respond(self, prompt):
        self.prompts.append(prompt)
        response = self.responses.pop(0)
        return self.response_model(grade=response)


class _FakeJevClient:
    instances = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        self.calls = []
        self.answer = SimpleNamespace(
            choice="2", probabilities={"2": 0.93, "1": 0.05, "0": 0.02}
        )
        type(self).instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        return SimpleNamespace(choices={"grade": self.answer})


def test_llm_engine_formats_prompt_once_and_normalizes_model():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pointwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMJudgeEngine(
            model="gpt-5-mini",
            system_prompt="Judge carefully.",
            rubric=RUBRIC,
        )
        _FakeAutoEnricher.instances[0].responses.append(1)

        judgment = engine.judge(
            prompt="Query: desk\nTitle: Study desk",
            question="Choose the best grade.",
            rubric=RUBRIC,
            prompt_has_question=False,
            prompt_has_rubric=False,
        )

    assert judgment.grade == 1
    assert _FakeAutoEnricher.instances[0].model == "openai/gpt-5-mini"
    assert _FakeAutoEnricher.instances[0].prompts == [
        "Query: desk\nTitle: Study desk\n\nChoose the best grade.\n\n"
        "2: relevant\n1: partially relevant\n0: irrelevant"
    ]


def test_llm_engine_does_not_duplicate_explicit_placeholders():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pointwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMJudgeEngine(
            model="openai/gpt-5-mini", system_prompt="Judge.", rubric=RUBRIC
        )
        _FakeAutoEnricher.instances[0].responses.append(2)
        engine.judge(
            prompt="Question: Choose.\nRubric:\n2: relevant\n1: partial\n0: no",
            question="Choose.",
            rubric=RUBRIC,
            prompt_has_question=True,
            prompt_has_rubric=True,
        )

    assert _FakeAutoEnricher.instances[0].prompts == [
        "Question: Choose.\nRubric:\n2: relevant\n1: partial\n0: no"
    ]


def test_llm_engine_appends_only_the_missing_rubric():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pointwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMJudgeEngine(
            model="gpt-5-mini", system_prompt="Judge.", rubric=RUBRIC
        )
        _FakeAutoEnricher.instances[0].responses.append(2)
        engine.judge(
            prompt="Question: Choose.",
            question="Choose.",
            rubric=RUBRIC,
            prompt_has_question=True,
            prompt_has_rubric=False,
        )

    assert _FakeAutoEnricher.instances[0].prompts == [
        "Question: Choose.\n\n2: relevant\n1: partially relevant\n0: irrelevant"
    ]


def test_llm_engine_no_cache_uses_underlying_enricher():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pointwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMJudgeEngine(
            model="gpt-5-mini", system_prompt="Judge.", rubric=RUBRIC, no_cache=True
        )
        fake = _FakeAutoEnricher.instances[0]
        fake.responses.append(0)
        judgment = engine.judge(
            prompt="state",
            question="question",
            rubric=RUBRIC,
            prompt_has_question=False,
            prompt_has_rubric=False,
        )

    assert judgment.grade == 0
    assert not fake.cached_calls
    assert len(fake.uncached_calls) == 1


def test_llm_engine_rejects_grade_outside_rubric():
    _FakeAutoEnricher.instances = []
    with patch("exps.judge.engines.llm_pointwise.AutoEnricher", _FakeAutoEnricher):
        engine = LLMJudgeEngine(
            model="gpt-5-mini", system_prompt="Judge.", rubric=RUBRIC
        )
        _FakeAutoEnricher.instances[0].responses.append(3)
        with pytest.raises(ValueError):
            engine.judge(
                prompt="state",
                question="question",
                rubric=RUBRIC,
                prompt_has_question=False,
                prompt_has_rubric=False,
            )


def test_jev_engine_sends_prompt_question_and_rubric_to_distinct_api_fields():
    _FakeJevClient.instances = []
    with (
        patch("exps.judge.engines.jev_pointwise.TypeSafeClient", _FakeJevClient),
        patch("exps.judge.engines.jev_pointwise.key_for_provider", return_value="key"),
    ):
        engine = JevJudgeEngine(model="jev/jev-latest", rubric=RUBRIC)
        judgment = engine.judge(
            prompt="Query: desk\nTitle: Study desk",
            question="Choose the relevance grade.",
            rubric=RUBRIC,
        )

    fake = _FakeJevClient.instances[0]
    assert fake.model == "jev-latest"
    assert fake.api_key == "key"
    assert judgment.grade == 2
    assert judgment.probability == 0.93
    state, questions = fake.calls[0]
    assert state == "Query: desk\nTitle: Study desk"
    assert questions["grade"].instructions == "Choose the relevance grade."
    assert questions["grade"].criteria == {
        "2": "relevant",
        "1": "partially relevant",
        "0": "irrelevant",
    }


def test_jev_engine_rejects_unknown_choice():
    _FakeJevClient.instances = []
    with (
        patch("exps.judge.engines.jev_pointwise.TypeSafeClient", _FakeJevClient),
        patch("exps.judge.engines.jev_pointwise.key_for_provider", return_value="key"),
    ):
        engine = JevJudgeEngine(model="jev/jev-latest", rubric=RUBRIC)
        _FakeJevClient.instances[0].answer = SimpleNamespace(
            choice="7", probabilities={"7": 1.0}
        )
        with pytest.raises(ValueError, match="rubric"):
            engine.judge(prompt="state", question="question", rubric=RUBRIC)
