from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from exps.agentic import conditions
from exps.agentic.conditions.base import ConditionContext, ConditionResult
from exps.agentic.conditions.simple import IterationsCondition, NumResultsCondition
from exps.bag_of_decisions.decision_question import DecisionQuestion
from tests.utils.agent_fakes import FakeJevChoice, FakeJevClient


_patch_jev_api = patch.multiple(
    "exps.agentic.conditions.jev_judge",
    TypeSafeClient=FakeJevClient,
    Choice=FakeJevChoice,
    key_for_provider=lambda _provider: "test-key",
    create=True,
)


def _condition_config(params: dict | None = None, *, prompt: str = "Improve the results."):
    configured_params = {
        "model": "jev/jev-latest",
        "probability_threshold": 0.7,
        "confidence_threshold": 0.6,
        "choices": {
            "Relevant": "The result satisfies the query.",
            "Neutral": "The result is related but does not satisfy the query.",
            "Irrelevant": "The result is unrelated to the query.",
        },
        "judge_prompt": "Query: {query}\nResult: {results}",
    }
    if params is not None:
        configured_params.update(params)
    return [
        {
            "jev_judge_relevance": {
                "prompt": prompt,
                "params": configured_params,
            }
        }
    ]


def _normalize(params: dict | None = None, *, kind: str = "validator"):
    return conditions.normalize_conditions(_condition_config(params), kind=kind)[0]


def _ranked_response(doc_ids: list[str] | None = None):
    return SimpleNamespace(
        output_parsed=SimpleNamespace(
            ranked_results=["101", "202", "303"] if doc_ids is None else doc_ids
        )
    )


def _evaluate(outcomes, *, params: dict | None = None, runs: int = 0, doc_ids=None):
    FakeJevClient.reset(
        [
            (
                outcome.get("choice"),
                (outcome.get("probabilities") or {}).get(outcome.get("choice")),
                outcome.get("confidence"),
            )
            for outcome in outcomes
        ]
    )

    condition = _normalize(params)
    agent_state = {"jev_judge_runs": runs}
    result = conditions.evaluate_validator(
        condition,
        num_loops=runs + 1,
        tool_calls=0,
        resp=_ranked_response(doc_ids),
        query="blue chair",
        corpus=pd.DataFrame(
            {
                "doc_id": [101, 202, 303],
                "title": ["Blue chair", "Red shoes", "Desk lamp"],
                "description": ["A blue seat", "Leather footwear", "A task light"],
            }
        ),
        lookup=None,
        judgments=None,
        agent_state=agent_state,
        logger=None,
    )
    return result, agent_state, FakeJevClient, FakeJevChoice


def _outcome(label="Relevant", probability=0.9, confidence=0.9):
    return {
        "choice": label,
        "probabilities": {label: probability},
        "confidence": confidence,
    }


def test_normalize_conditions_builds_typed_conditions_with_shared_result_contract():
    stop = conditions.normalize_conditions(
        [{"iterations": {"prompt": "Keep going", "params": {"iterations": 2}}}],
        kind="stop",
    )[0]
    validator = conditions.normalize_conditions(
        [{"num_results": {"prompt": "Need results", "params": {"min_results": 3}}}],
        kind="validator",
    )[0]

    assert isinstance(stop, IterationsCondition)
    assert isinstance(validator, NumResultsCondition)
    assert stop.evaluate(
        ConditionContext(
            num_loops=1,
            tool_calls=0,
            response=None,
        )
    ) == ConditionResult.unsatisfied("Keep going")


def test_normalize_jev_judge_relevance_defaults_and_normalizes_thresholds():
    condition = _normalize({"probability_threshold": "0.75", "confidence_threshold": 0.8})

    assert condition["name"] == "jev_judge_relevance"
    assert condition["params"]["probability_threshold"] == 0.75
    assert condition["params"]["confidence_threshold"] == 0.8
    assert condition["params"]["max_runs"] == 2
    assert condition["params"]["choices"]["Relevant"] == "The result satisfies the query."


@pytest.mark.parametrize(
    "missing_param",
    [
        "model",
        "probability_threshold",
        "confidence_threshold",
        "choices",
        "judge_prompt",
    ],
)
def test_normalize_jev_judge_relevance_requires_all_parameters(missing_param):
    config = _condition_config()
    params = config[0]["jev_judge_relevance"]["params"]
    params.pop(missing_param)

    with pytest.raises(ValueError, match=rf"requires params\.{missing_param}"):
        conditions.normalize_conditions(config, kind="validator")


@pytest.mark.parametrize("threshold_name", ["probability_threshold", "confidence_threshold"])
@pytest.mark.parametrize("invalid_threshold", [-0.01, 1.01, float("nan"), float("inf"), True, "bad"])
def test_normalize_jev_judge_relevance_rejects_invalid_thresholds(
    threshold_name, invalid_threshold
):
    with pytest.raises(ValueError, match=threshold_name):
        _normalize({threshold_name: invalid_threshold})


@pytest.mark.parametrize(
    "params, message",
    [
        ({"model": "openai/gpt-5-mini"}, "Jev model"),
        ({"model": "jev/"}, "Jev model"),
        ({"choices": {}}, "non-empty mapping"),
        ({"choices": ["Relevant", "Irrelevant"]}, "non-empty mapping"),
        ({"choices": {" ": "No label"}}, "labels must be non-empty"),
        ({"choices": {"Relevant": "  "}}, "requires non-empty criteria"),
        ({"judge_prompt": "  "}, "non-empty params.judge_prompt"),
        ({"max_runs": 0}, "max_runs > 0"),
        ({"max_runs": 1.5}, "max_runs > 0"),
        ({"max_runs": float("inf")}, "max_runs > 0"),
        ({"max_runs": True}, "max_runs > 0"),
        ({"max_runs": "many"}, "max_runs > 0"),
    ],
)
def test_normalize_jev_judge_relevance_rejects_malformed_configuration(params, message):
    with pytest.raises(ValueError, match=message):
        _normalize(params)


def test_jev_judge_relevance_is_validator_only():
    with pytest.raises(ValueError, match="only supported for validators"):
        _normalize(kind="stop")


@_patch_jev_api
def test_jev_judge_relevance_accepts_all_results_when_both_scores_clear_thresholds():
    result, agent_state, fake_client, fake_choice = _evaluate(
        [_outcome(), _outcome(), _outcome()],
    )

    assert result is True
    assert agent_state["jev_judge_runs"] == 1
    assert len(fake_client.calls) == 3
    assert fake_client.clients[0].model == "jev-latest"
    assert fake_client.clients[0].api_key == "test-key"
    _, choice = fake_client.calls[0]
    assert isinstance(choice, fake_choice)
    assert choice.criteria == _condition_config()[0]["jev_judge_relevance"]["params"]["choices"]
    assert "blue chair" in choice.instructions
    assert "Blue chair" in choice.instructions


@_patch_jev_api
@pytest.mark.parametrize(
    "probability, confidence",
    [
        (0.7, 0.9),  # Probability threshold is strict.
        (0.9, 0.6),  # Confidence threshold is strict.
        (0.69, 0.59),
        (1.01, 0.9),
        (float("inf"), 0.9),
        (None, 0.9),
        (0.9, None),
        (float("nan"), 0.9),
        (0.9, float("nan")),
        (0.9, float("inf")),
    ],
)
def test_jev_judge_relevance_retries_if_either_threshold_is_not_cleared(
    probability, confidence
):
    outcome = {
        "choice": "Relevant",
        "probabilities": {"Relevant": probability} if probability is not None else None,
        "confidence": confidence,
    }

    result, agent_state, _, _ = _evaluate(
        [outcome, outcome, outcome],
    )

    assert isinstance(result, str)
    assert "Improve the results." in result
    assert "uncertain" in result.lower() or "Relevant" in result
    assert agent_state["jev_judge_runs"] == 1


@_patch_jev_api
@pytest.mark.parametrize("label", ["Neutral", "Irrelevant"])
def test_jev_judge_relevance_retries_on_accepted_non_relevant_labels(label):
    result, _, _, _ = _evaluate(
        [_outcome(label), _outcome(), _outcome()],
    )

    assert isinstance(result, str)
    assert label in result


@_patch_jev_api
def test_jev_judge_relevance_retries_if_jev_returns_an_unconfigured_label():
    result, _, _, _ = _evaluate(
        [_outcome("Unexpected"), _outcome(), _outcome()],
    )

    assert isinstance(result, str)
    assert "Unexpected" in result or "uncertain" in result.lower()


@_patch_jev_api
def test_jev_judge_relevance_accepts_after_max_runs_to_avoid_infinite_retries():
    result, agent_state, fake_client, _ = _evaluate(
        [_outcome("Irrelevant"), _outcome("Irrelevant"), _outcome("Irrelevant")],
        params={"max_runs": 1},
    )

    assert result is True
    assert agent_state["jev_judge_runs"] == 1
    assert len(fake_client.calls) == 3


@_patch_jev_api
def test_jev_judge_relevance_does_not_pass_empty_result_lists():
    result, _, _, _ = _evaluate([], doc_ids=[])

    assert isinstance(result, str)


@_patch_jev_api
def test_jev_judge_relevance_marks_unrenderable_results_uncertain():
    result, _, fake_client, _ = _evaluate([], doc_ids=["not-a-document-id"])

    assert isinstance(result, str)
    assert "uncertain" in result.lower()
    assert fake_client.calls == []


def _bag_of_decisions_condition_config(params: dict | None = None):
    configured_params = {
        "model": "jev/jev-latest",
        "positive_probability_threshold": 0.75,
        "negative_probability_threshold": 0.25,
        "max_runs": 2,
        "generator": {
            "model": "gpt-5-mini",
            "system_prompt": "Generate relevance criteria.",
            "prompt": "Generate criteria for {query}.",
        },
        "state_format": "Query: {query}\n{title}\n{description}",
    }
    if params is not None:
        configured_params.update(params)
    return [
        {
            "jev_bag_of_decisions_judge": {
                "prompt": "Improve the results.",
                "params": configured_params,
            }
        }
    ]


class _ScriptedDecisionGenerator:
    instances = []
    questions = [
        "Does this product satisfy the query?",
        "Is this product intended for the requested audience?",
        "Does this product match the requested use?",
    ]

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.generate_calls = []
        self.__class__.instances.append(self)

    def generate(self, query):
        self.generate_calls.append(query)
        return [DecisionQuestion(instructions=q) for q in self.questions]


class _ScriptedNoul:
    def __init__(self, *, instructions, criteria=None):
        self.instructions = instructions
        self.criteria = criteria


class _ScriptedNoulClient:
    instances = []
    responses = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        self.calls = []
        self.__class__.instances.append(self)

    def system_one(self, *, state, questions, retry=None):
        self.calls.append((state, questions))
        probabilities = self.responses.pop(0)
        answers = {
            question_id: SimpleNamespace(noul=probabilities[index])
            for index, question_id in enumerate(questions)
        }
        return SimpleNamespace(answers=answers)


def _evaluate_bag_of_decisions(
    monkeypatch,
    probabilities,
    *,
    agent_state=None,
    doc_ids=None,
    params=None,
    logger=None,
):
    if agent_state is None:
        _ScriptedDecisionGenerator.instances = []
        _ScriptedNoulClient.instances = []
    _ScriptedNoulClient.responses = list(probabilities)
    monkeypatch.setattr(
        conditions.jev_bag_of_decisions,
        "DecisionGenerator",
        _ScriptedDecisionGenerator,
        raising=False,
    )
    monkeypatch.setattr(
        conditions.jev_bag_of_decisions, "Noul", _ScriptedNoul, raising=False
    )
    monkeypatch.setattr(
        conditions.jev_bag_of_decisions, "TypeSafeClient", _ScriptedNoulClient
    )
    monkeypatch.setattr(
        conditions.jev_bag_of_decisions,
        "key_for_provider",
        lambda _provider: "test-key",
    )
    condition = conditions.normalize_conditions(
        _bag_of_decisions_condition_config(params), kind="validator"
    )[0]
    current_state = agent_state if agent_state is not None else {}
    result = conditions.evaluate_validator(
        condition,
        num_loops=current_state.get("jev_bag_of_decisions_judge_runs", 0) + 1,
        tool_calls=0,
        resp=_ranked_response(["101"] if doc_ids is None else doc_ids),
        query="made for kids",
        corpus=pd.DataFrame(
            {
                "doc_id": [101, 202, 303],
                "title": ["Kids craft table", "Red shoes", "Task lamp"],
                "description": [
                    "A small activity table for children.",
                    "Leather footwear.",
                    "An adjustable desk light.",
                ],
            }
        ),
        lookup=None,
        judgments=None,
        agent_state=current_state,
        logger=logger,
    )
    return result, current_state


def test_normalize_jev_bag_of_decisions_judge_config():
    condition = conditions.normalize_conditions(
        _bag_of_decisions_condition_config(
            {
                "positive_probability_threshold": "0.8",
                "negative_probability_threshold": "0.2",
            }
        ),
        kind="validator",
    )[0]

    assert condition["name"] == "jev_bag_of_decisions_judge"
    assert condition["params"]["positive_probability_threshold"] == 0.8
    assert condition["params"]["negative_probability_threshold"] == 0.2
    assert condition["params"]["max_runs"] == 2


@pytest.mark.parametrize(
    "params, message",
    [
        ({"positive_probability_threshold": 0.2}, "negative_probability_threshold <"),
        ({"negative_probability_threshold": 0.75}, "negative_probability_threshold <"),
        ({"positive_probability_threshold": 1.01}, "positive_probability_threshold"),
        ({"negative_probability_threshold": -0.01}, "negative_probability_threshold"),
        ({"model": "openai/gpt-5-mini"}, "Jev model"),
        ({"generator": None}, "params.generator as a mapping"),
        ({"max_runs": 0}, "max_runs > 0"),
    ],
)
def test_normalize_jev_bag_of_decisions_judge_rejects_invalid_config(params, message):
    with pytest.raises(ValueError, match=message):
        conditions.normalize_conditions(
            _bag_of_decisions_condition_config(params), kind="validator"
        )


def test_jev_bag_of_decisions_judge_is_validator_only():
    with pytest.raises(ValueError, match="only supported for validators"):
        conditions.normalize_conditions(
            _bag_of_decisions_condition_config(), kind="stop"
        )


def test_jev_bag_of_decisions_judge_scores_and_emits_only_thresholded_questions(
    monkeypatch,
):
    result, agent_state = _evaluate_bag_of_decisions(
        monkeypatch,
        [[0.75, 0.25, 0.5]],
    )

    assert isinstance(result, str)
    assert "Jev bag-of-decisions evaluations" in result
    assert "👍 Does this product satisfy the query?" in result
    assert "👎 Is this product intended for the requested audience?" in result
    assert "Does this product match the requested use?" not in result
    assert "1.500" in result
    assert agent_state["jev_bag_of_decisions_judge_runs"] == 1
    assert len(_ScriptedNoulClient.instances[0].calls) == 1
    state, questions = _ScriptedNoulClient.instances[0].calls[0]
    assert "made for kids" in state
    assert "Kids craft table" in state
    assert len(questions) == 3


def test_jev_bag_of_decisions_judge_logs_per_document_progress(monkeypatch):
    class CaptureLogger:
        def __init__(self):
            self.messages = []

        def info(self, message, *args):
            self.messages.append(message % args if args else message)

    logger = CaptureLogger()
    result, _ = _evaluate_bag_of_decisions(
        monkeypatch,
        [[0.9, 0.5, 0.1]],
        logger=logger,
    )

    assert isinstance(result, str)
    assert any("agentic_jev_rubric_start" in message for message in logger.messages)
    assert any("agentic_jev_rubric_complete" in message for message in logger.messages)
    assert any("agentic_jev_document_start" in message for message in logger.messages)
    assert any("agentic_jev_request_start" in message for message in logger.messages)
    assert any("agentic_jev_request_complete" in message for message in logger.messages)
    assert any("agentic_jev_document_complete" in message for message in logger.messages)


def test_jev_bag_of_decisions_judge_passes_results_with_positive_only_evidence(
    monkeypatch,
):
    result, _ = _evaluate_bag_of_decisions(
        monkeypatch,
        [[0.9, 0.5, 0.75]],
    )

    assert result is True


def test_jev_bag_of_decisions_judge_accepts_after_max_runs(monkeypatch):
    result, agent_state = _evaluate_bag_of_decisions(
        monkeypatch,
        [[0.1, 0.1, 0.1]],
        params={"max_runs": 1},
    )

    assert result is True
    assert agent_state["jev_bag_of_decisions_judge_runs"] == 1


def test_jev_bag_of_decisions_judge_reuses_generated_rubric_across_retries(
    monkeypatch,
):
    _ScriptedDecisionGenerator.instances = []
    _ScriptedNoulClient.instances = []
    state = {}
    first_result, _ = _evaluate_bag_of_decisions(
        monkeypatch,
        [[0.1, 0.1, 0.1]],
        agent_state=state,
    )
    assert isinstance(first_result, str)

    second_result, _ = _evaluate_bag_of_decisions(
        monkeypatch,
        [[0.9, 0.9, 0.9]],
        agent_state=state,
    )

    assert second_result is True
    assert sum(
        len(generator.generate_calls)
        for generator in _ScriptedDecisionGenerator.instances
    ) == 1
