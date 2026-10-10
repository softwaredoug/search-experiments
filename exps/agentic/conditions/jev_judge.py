from __future__ import annotations

import math
from dataclasses import dataclass

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Choice, TypeSafeClient

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


def _jev_model_name(model: str) -> str:
    provider, separator, model_name = model.partition("/")
    if provider.lower() == "jev":
        return model_name if separator else "jev-latest"
    return model


def _is_valid_score(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _run_jev_judge(
    *,
    query: str,
    corpus,
    lookup: dict | None,
    ranked_doc_ids: list[str],
    model: str,
    choices: dict[str, str],
    probability_threshold: float,
    confidence_threshold: float,
    judge_prompt: str,
) -> list[dict]:
    if not ranked_doc_ids:
        return []
    client = TypeSafeClient(
        api_key=key_for_provider("typesafe"),
        model=_jev_model_name(model),
    )
    evaluations = []
    for doc_id in ranked_doc_ids:
        result_block = judging._render_results_for_judge(
            corpus=corpus,
            ranked_doc_ids=[str(doc_id)],
            lookup=lookup,
        )
        row, _ = judging._judge_row_for_doc_id(
            corpus=corpus, doc_id=doc_id, lookup=lookup
        )
        title = str(row.get("title", "")) if row is not None else ""
        if not result_block:
            evaluations.append(
                {
                    "doc_id": str(doc_id),
                    "title": title,
                    "label": None,
                    "probability": None,
                    "confidence": None,
                    "accepted": False,
                }
            )
            continue
        instructions = judge_prompt.format(query=query, results=result_block)
        response = client.system_one(
            state=query,
            questions={
                "relevance": Choice(instructions=instructions, criteria=choices)
            },
        )
        answers = getattr(response, "choices", None)
        answer = answers.get("relevance") if isinstance(answers, dict) else None
        label = getattr(answer, "choice", None)
        confidence = getattr(answer, "confidence", None)
        probabilities = getattr(answer, "probabilities", None)
        probability = (
            probabilities.get(label)
            if isinstance(probabilities, dict) and isinstance(label, str)
            else None
        )
        accepted = (
            isinstance(label, str)
            and label in choices
            and _is_valid_score(probability)
            and probability > probability_threshold
            and _is_valid_score(confidence)
            and confidence > confidence_threshold
        )
        evaluations.append(
            {
                "doc_id": str(doc_id),
                "title": title,
                "label": label,
                "probability": probability,
                "confidence": confidence,
                "accepted": accepted,
            }
        )
    return evaluations


def _jev_judge_is_passing(evaluations: list[dict]) -> bool:
    return bool(evaluations) and all(
        evaluation["accepted"] and evaluation["label"] == "Relevant"
        for evaluation in evaluations
    )


def _jev_judge_feedback(evaluations: list[dict]) -> str:
    lines = []
    for idx, evaluation in enumerate(evaluations, start=1):
        label = evaluation["label"]
        if not evaluation["accepted"]:
            label = f"uncertain {label or 'unknown'}"
        probability = evaluation["probability"]
        confidence = evaluation["confidence"]
        probability_text = f"{probability:.3f}" if _is_valid_score(probability) else "n/a"
        confidence_text = f"{confidence:.3f}" if _is_valid_score(confidence) else "n/a"
        title = f" {evaluation['title']}" if evaluation.get("title") else ""
        lines.append(
            f"{idx}. {label} (probability: {probability_text}, "
            f"confidence: {confidence_text}){title} (ID: {evaluation['doc_id']})"
        )
    return "\n".join(lines)


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

        evaluations = _run_jev_judge(
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
        if _jev_judge_is_passing(evaluations):
            return ConditionResult.success()
        if judge_runs >= int(params.get("max_runs", 2)):
            return ConditionResult.success()

        feedback = _jev_judge_feedback(evaluations)
        return self.feedback(
            f"{self.prompt}\n\nJev evaluations:\n\n{feedback}\n\n"
            "System reminder: return DOC IDs ranked best to worst."
        )
