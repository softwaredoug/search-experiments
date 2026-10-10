from __future__ import annotations

import math
from dataclasses import dataclass

from exps.agentic.conditions import judging
from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
    ranked_doc_ids,
)


def _parse_threshold(value, *, name: str, condition_name: str) -> float:
    message = f"Condition '{condition_name}' requires params.{name} between 0 and 1."
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError(message)
    return threshold


def _positive_integer(value, *, name: str) -> int:
    try:
        if isinstance(value, bool):
            raise ValueError
        result = int(value)
        if str(value).strip() not in {str(result), f"{result}.0"} or result <= 0:
            raise ValueError
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"Condition '{name}' requires params.max_runs > 0.") from exc
    return result


@dataclass
class JevJudgeRelevance(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> JevJudgeRelevance:
        name = "jev_judge_relevance"
        if kind != "validator":
            raise ValueError(f"{name} is only supported for validators.")
        for key in (
            "model",
            "probability_threshold",
            "confidence_threshold",
            "choices",
            "judge_prompt",
        ):
            if key not in params:
                raise ValueError(f"Condition '{name}' requires params.{key}.")
        model = params["model"]
        if (
            not isinstance(model, str)
            or model.strip() != model
            or model.split("/", 1)[0].lower() != "jev"
            or ("/" in model and not model.split("/", 1)[1].strip())
        ):
            raise ValueError(f"Condition '{name}' requires a Jev model (jev/*).")
        if not isinstance(params["judge_prompt"], str) or not params[
            "judge_prompt"
        ].strip():
            raise ValueError(
                f"Condition '{name}' requires a non-empty params.judge_prompt."
            )
        choices = params["choices"]
        if not isinstance(choices, dict) or not choices:
            raise ValueError(
                f"Condition '{name}' requires params.choices as a non-empty mapping."
            )
        for label, criteria in choices.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("Jev judge choice labels must be non-empty strings.")
            if not isinstance(criteria, str) or not criteria.strip():
                raise ValueError(
                    f"Jev judge choice {label!r} requires non-empty criteria."
                )
        params["probability_threshold"] = _parse_threshold(
            params["probability_threshold"],
            name="probability_threshold",
            condition_name=name,
        )
        params["confidence_threshold"] = _parse_threshold(
            params["confidence_threshold"],
            name="confidence_threshold",
            condition_name=name,
        )
        params["max_runs"] = _positive_integer(
            params.get("max_runs", 2), name=name
        )
        return cls(name=name, prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        params = self.params
        if context.agent_state is not None:
            run_key = "jev_judge_runs"
            context.agent_state[run_key] = context.agent_state.get(run_key, 0) + 1
            judge_runs = context.agent_state[run_key]
        else:
            judge_runs = context.num_loops

        evaluations = judging._run_jev_judge(
            query=context.query,
            corpus=context.corpus,
            lookup=context.lookup,
            ranked_doc_ids=ranked_doc_ids(context.response),
            model=str(params["model"]),
            choices=params["choices"],
            probability_threshold=params["probability_threshold"],
            confidence_threshold=params["confidence_threshold"],
            judge_prompt=str(params["judge_prompt"]),
        )
        if judging._jev_judge_is_passing(evaluations):
            return ConditionResult.success()
        if judge_runs >= int(params.get("max_runs", 2)):
            return ConditionResult.success()

        feedback = judging._jev_judge_feedback(evaluations)
        return self.feedback(
            f"{self.prompt}\n\nJev evaluations:\n\n{feedback}\n\n"
            "System reminder: return DOC IDs ranked best to worst."
        )
