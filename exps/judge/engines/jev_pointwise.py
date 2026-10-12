from __future__ import annotations

import math
from collections.abc import Mapping

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Choice, TypeSafeClient

from exps.judge.engines.common import (
    JudgeResult,
    choice_grade,
    normalize_rubric,
    rubric_criteria,
)


def _jev_model_name(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("strategy.params.model must be a non-empty string.")
    provider, separator, model_name = model.strip().partition("/")
    if provider.lower() != "jev":
        raise ValueError("JevJudgeEngine requires a Jev model (jev/*).")
    if not separator:
        return "jev-latest"
    if not model_name.strip():
        raise ValueError("Jev model name must not be empty.")
    return model_name.strip()


def _answer_for(response, question_id: str):
    for attribute in ("choices", "answers"):
        answers = getattr(response, attribute, None)
        if isinstance(answers, Mapping) and question_id in answers:
            return answers[question_id]
    return None


class JevJudgeEngine:
    """Use Jev Choice to select a numeric grade from the configured rubric."""

    def __init__(self, *, model: str, rubric: Mapping):
        self.model = _jev_model_name(model)
        self.rubric = normalize_rubric(rubric)
        self.client = TypeSafeClient(
            api_key=key_for_provider("typesafe"),
            model=self.model,
        )

    def judge(self, *, prompt: str, question: str, rubric: Mapping, **_kwargs) -> JudgeResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Judge prompt must be a non-empty string.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Judge question must be a non-empty string.")
        active_rubric = normalize_rubric(rubric)
        if active_rubric != self.rubric:
            raise ValueError("Judge request rubric differs from the configured rubric.")
        question_id = "grade"
        response = self.client.system_one(
            state=prompt,
            questions={
                question_id: Choice(
                    instructions=question,
                    criteria=rubric_criteria(self.rubric),
                )
            },
        )
        answer = _answer_for(response, question_id)
        choice = getattr(answer, "choice", None) if answer is not None else None
        grade = choice_grade(choice, self.rubric)
        probabilities = getattr(answer, "probabilities", None)
        probability = None
        if isinstance(probabilities, Mapping):
            probability = probabilities.get(f"{grade:g}", probabilities.get(choice))
        if probability is not None:
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Jev returned an invalid probability for its choice.")
            probability = float(probability)
        return JudgeResult(grade=grade, probability=probability)
