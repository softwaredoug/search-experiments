from __future__ import annotations

import math
from collections.abc import Mapping

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Choice, TypeSafeClient

from exps.judge.engines.common import PairwiseJudgeResult
from exps.judge.engines.jev_pointwise import _answer_for, _jev_model_name


class JevPairwiseJudgeEngine:
    """Use a Jev Choice to select the more relevant of two documents."""

    def __init__(self, *, model: str, explicit_neither: bool = False):
        if not isinstance(explicit_neither, bool):
            raise ValueError("strategy.params.explicit_neither must be a boolean.")
        self.model = _jev_model_name(model)
        self.explicit_neither = explicit_neither
        criteria = {
            "lhs": "The left-hand document is more relevant to the query.",
            "rhs": "The right-hand document is more relevant to the query.",
        }
        if explicit_neither:
            criteria["neither"] = "Neither document is more relevant than the other."
        self.criteria = criteria
        self.client = TypeSafeClient(
            api_key=key_for_provider("typesafe"),
            model=self.model,
        )

    def judge(self, *, prompt: str, question: str, **_kwargs) -> PairwiseJudgeResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Judge prompt must be a non-empty string.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Judge question must be a non-empty string.")
        question_id = "winner"
        response = self.client.system_one(
            state=prompt,
            questions={
                question_id: Choice(
                    instructions=question,
                    criteria=self.criteria,
                )
            },
        )
        answer = _answer_for(response, question_id)
        outcome = getattr(answer, "choice", None) if answer is not None else None
        allowed = {"lhs", "rhs"}
        if self.explicit_neither:
            allowed.add("neither")
        if outcome not in allowed:
            raise ValueError(f"Jev returned an invalid pairwise outcome: {outcome!r}.")
        probability = None
        probabilities = getattr(answer, "probabilities", None)
        if isinstance(probabilities, Mapping):
            probability = probabilities.get(outcome)
        if probability is not None:
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Jev returned an invalid probability for its choice.")
            probability = float(probability)
        return PairwiseJudgeResult(outcome=outcome, probability=probability)
