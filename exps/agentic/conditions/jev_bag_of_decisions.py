from __future__ import annotations

import hashlib
import json
import math
import string
from dataclasses import dataclass
from time import monotonic
from typing import Any

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Noul, NoulCriteria, RetryPolicy, TypeSafeClient

from exps.agentic.conditions import judging
from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
    ranked_doc_ids,
)
from exps.agentic.tracing import set_trace_stage, trace_event
from exps.bag_of_decisions.decision_generator import DecisionGenerator
from exps.bag_of_decisions.decision_question import DecisionQuestion
from exps.agentic.conditions.jev_judge import _parse_threshold, _positive_integer


class _TracingRetryPolicy(RetryPolicy):
    """Keep TypeSafe's defaults and expose retry attempts in agent traces."""

    def __init__(self, *, logger, context: dict[str, Any]):
        super().__init__()
        object.__setattr__(self, "_trace_logger", logger)
        object.__setattr__(self, "_trace_context", context)

    def _retryable(self, error: BaseException) -> bool:
        retryable = super()._retryable(error)
        failure_number = self._trace_context.get("failure_number", 0) + 1
        self._trace_context["failure_number"] = failure_number
        trace_event(
            self._trace_logger,
            "agentic_jev_request_retry_check",
            doc_index=self._trace_context["doc_index"],
            total_docs=self._trace_context["total_docs"],
            doc_id=self._trace_context["doc_id"],
            failure_number=failure_number,
            retrying=retryable and failure_number <= self.max_retries,
            error_type=type(error).__name__,
            status_code=getattr(error, "status", None),
            request_id=getattr(error, "request_id", None),
        )
        return retryable


def _jev_model_name(model: str) -> str:
    provider, separator, model_name = model.partition("/")
    if provider.lower() == "jev":
        return model_name if separator else "jev-latest"
    return model


def _is_valid_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _run_jev_bag_of_decisions_judge(
    *,
    query: str,
    corpus,
    lookup: dict | None,
    ranked_doc_ids: list[str],
    model: str,
    generator: dict[str, Any],
    positive_probability_threshold: float,
    negative_probability_threshold: float,
    state_format: str,
    agent_state: dict | None,
    logger,
) -> list[dict[str, Any]]:
    """Score ranked documents against a query-specific generated rubric."""
    if not ranked_doc_ids:
        return []

    rubric_key_payload = json.dumps(
        {"query": query, "generator": generator},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    rubric_key = hashlib.sha256(rubric_key_payload).hexdigest()
    rubric_cache = None
    if agent_state is not None:
        rubric_cache = agent_state.setdefault(
            "jev_bag_of_decisions_judge_rubrics", {}
        )
        if not isinstance(rubric_cache, dict):
            rubric_cache = {}
            agent_state["jev_bag_of_decisions_judge_rubrics"] = rubric_cache

    if rubric_cache is not None and rubric_key in rubric_cache:
        raw_questions = rubric_cache[rubric_key]
        decisions = [
            DecisionQuestion(
                instructions=item["instructions"],
                criteria=item.get("criteria"),
            )
            for item in raw_questions
            if isinstance(item, dict)
            and isinstance(item.get("instructions"), str)
        ]
        trace_event(
            logger,
            "agentic_jev_rubric_reused",
            query=query,
            question_count=len(decisions),
        )
    else:
        generator_options = {
            key: generator[key]
            for key in (
                "model",
                "system_prompt",
                "prompt",
                "reasoning",
                "temperature",
                "verbosity",
                "no_cache",
            )
            if key in generator
        }
        rubric_started = monotonic()
        set_trace_stage(
            agent_state,
            "jev_rubric_generation",
            generator_model=generator_options.get("model"),
        )
        trace_event(
            logger,
            "agentic_jev_rubric_start",
            generator_model=generator_options.get("model"),
        )
        try:
            question_generator = DecisionGenerator(**generator_options)
            decisions = question_generator.generate(query)
        except Exception as exc:
            trace_event(
                logger,
                "agentic_jev_rubric_error",
                elapsed_seconds=round(monotonic() - rubric_started, 3),
                error_type=type(exc).__name__,
            )
            raise
        decisions = [
            decision
            for decision in decisions
            if isinstance(decision, DecisionQuestion)
            and isinstance(decision.instructions, str)
            and decision.instructions.strip()
        ]
        if rubric_cache is not None:
            rubric_cache[rubric_key] = [
                {
                    "instructions": decision.instructions,
                    "criteria": (
                        dict(decision.criteria)
                        if decision.criteria is not None
                        else None
                    ),
                }
                for decision in decisions
            ]
        trace_event(
            logger,
            "agentic_jev_rubric_complete",
            elapsed_seconds=round(monotonic() - rubric_started, 3),
            question_count=len(decisions),
        )

    if not decisions:
        return [
            {
                "doc_id": str(doc_id),
                "title": "",
                "score": None,
                "positive_decisions": [],
                "negative_decisions": [],
            }
            for doc_id in ranked_doc_ids
        ]

    client = TypeSafeClient(
        api_key=key_for_provider("typesafe"),
        model=_jev_model_name(model),
    )
    evaluations: list[dict[str, Any]] = []
    total_docs = len(ranked_doc_ids)
    for doc_index, doc_id in enumerate(ranked_doc_ids, start=1):
        document_started = monotonic()
        set_trace_stage(
            agent_state,
            "jev_document_evaluation",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(doc_id),
            question_count=len(decisions),
        )
        trace_event(
            logger,
            "agentic_jev_document_start",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(doc_id),
            question_count=len(decisions),
        )
        row, normalized_doc_id = judging._judge_row_for_doc_id(
            corpus=corpus,
            doc_id=doc_id,
            lookup=lookup,
        )
        title = str(row.get("title", "")) if row is not None else ""
        if row is None or normalized_doc_id is None:
            trace_event(
                logger,
                "agentic_jev_document_complete",
                doc_index=doc_index,
                total_docs=total_docs,
                doc_id=str(doc_id),
                elapsed_seconds=round(monotonic() - document_started, 3),
                outcome="document_not_found",
            )
            evaluations.append(
                {
                    "doc_id": str(doc_id),
                    "title": title,
                    "score": None,
                    "positive_decisions": [],
                    "negative_decisions": [],
                }
            )
            continue

        row_values = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        try:
            state = state_format.format_map({**row_values, "query": query})
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "Could not render params.state_format for a corpus row."
            ) from exc

        questions = {}
        for index, decision in enumerate(decisions):
            if decision.criteria is None:
                question = Noul(instructions=decision.instructions)
            else:
                question = Noul(
                    instructions=decision.instructions,
                    criteria=NoulCriteria(**decision.criteria),
                )
            questions[f"decision_{index}"] = question
        request_started = monotonic()
        trace_event(
            logger,
            "agentic_jev_request_start",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(normalized_doc_id),
            question_count=len(questions),
        )
        retry_policy = _TracingRetryPolicy(
            logger=logger,
            context={
                "doc_index": doc_index,
                "total_docs": total_docs,
                "doc_id": str(normalized_doc_id),
            },
        )
        try:
            response = client.system_one(
                state=state,
                questions=questions,
                retry=retry_policy,
            )
        except Exception as exc:
            trace_event(
                logger,
                "agentic_jev_request_error",
                doc_index=doc_index,
                total_docs=total_docs,
                doc_id=str(normalized_doc_id),
                elapsed_seconds=round(monotonic() - request_started, 3),
                error_type=type(exc).__name__,
            )
            raise
        answers = getattr(response, "answers", None)
        answers = answers if isinstance(answers, dict) else {}

        score = 0.0
        valid_answers = 0
        positive_decisions = []
        negative_decisions = []
        for index, decision in enumerate(decisions):
            answer = answers.get(f"decision_{index}")
            probability = getattr(answer, "noul", None)
            if not _is_valid_score(probability):
                continue
            probability = float(probability)
            score += probability
            valid_answers += 1
            item = {
                "question": decision.instructions,
                "probability": probability,
            }
            if probability >= positive_probability_threshold:
                positive_decisions.append(item)
            elif probability <= negative_probability_threshold:
                negative_decisions.append(item)

        evaluations.append(
            {
                "doc_id": str(normalized_doc_id),
                "title": title,
                "score": score if valid_answers else None,
                "positive_decisions": positive_decisions,
                "negative_decisions": negative_decisions,
            }
        )
        trace_event(
            logger,
            "agentic_jev_request_complete",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(normalized_doc_id),
            elapsed_seconds=round(monotonic() - request_started, 3),
            request_id=getattr(response, "request_id", None),
            valid_answers=valid_answers,
            score=score if valid_answers else None,
        )
        trace_event(
            logger,
            "agentic_jev_document_complete",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(normalized_doc_id),
            elapsed_seconds=round(monotonic() - document_started, 3),
            outcome="evaluated",
        )
    return evaluations


def _jev_bag_of_decisions_judge_is_passing(
    evaluations: list[dict[str, Any]],
) -> bool:
    if not evaluations:
        return False
    return all(
        evaluation.get("score") is not None
        and bool(evaluation.get("positive_decisions"))
        and not evaluation.get("negative_decisions")
        for evaluation in evaluations
    )


def _jev_bag_of_decisions_judge_feedback(
    evaluations: list[dict[str, Any]],
) -> str:
    lines = []
    for index, evaluation in enumerate(evaluations, start=1):
        title = f" {evaluation['title']}" if evaluation.get("title") else ""
        score = evaluation.get("score")
        score_text = (
            f"{score:.3f}"
            if isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and score >= 0
            else "n/a"
        )
        lines.append(
            f"{index}.{title} (ID: {evaluation['doc_id']}; "
            f"rubric score: {score_text})"
        )
        positive_decisions = evaluation.get("positive_decisions") or []
        negative_decisions = evaluation.get("negative_decisions") or []
        for decision in positive_decisions:
            lines.append(
                f"   👍 {decision['question']} "
                f"(P(yes)={decision['probability']:.3f})"
            )
        for decision in negative_decisions:
            lines.append(
                f"   👎 {decision['question']} "
                f"(P(yes)={decision['probability']:.3f})"
            )
        if not positive_decisions and not negative_decisions:
            lines.append("   No criteria cleared either probability threshold.")
    if not lines:
        return "No results were available to evaluate."
    return "\n".join(lines)


@dataclass
class JevBagOfDecisionsJudge(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> JevBagOfDecisionsJudge:
        name = "jev_bag_of_decisions_judge"
        if kind != "validator":
            raise ValueError(f"{name} is only supported for validators.")
        for key in (
            "model",
            "positive_probability_threshold",
            "negative_probability_threshold",
            "generator",
            "state_format",
        ):
            if key not in params:
                raise ValueError(
                    f"Condition '{name}' requires params.{key}."
                )
        params["positive_probability_threshold"] = _parse_threshold(
            params["positive_probability_threshold"],
            name="positive_probability_threshold",
            condition_name=name,
        )
        params["negative_probability_threshold"] = _parse_threshold(
            params["negative_probability_threshold"],
            name="negative_probability_threshold",
            condition_name=name,
        )
        if (
            params["negative_probability_threshold"]
            >= params["positive_probability_threshold"]
        ):
            raise ValueError(
                f"Condition '{name}' requires "
                "negative_probability_threshold < positive_probability_threshold."
            )
        generator = params["generator"]
        if not isinstance(generator, dict):
            raise ValueError(f"Condition '{name}' requires params.generator as a mapping.")
        for key in ("model", "system_prompt", "prompt"):
            if key not in generator:
                raise ValueError(f"Condition '{name}' requires params.generator.{key}.")
        for key in ("model", "state_format"):
            if not isinstance(params.get(key), str) or not params[key].strip():
                raise ValueError(f"Condition '{name}' requires a string {key}.")
        try:
            list(string.Formatter().parse(params["state_format"]))
        except ValueError as exc:
            raise ValueError(
                f"Condition '{name}' requires a valid params.state_format."
            ) from exc
        for key in ("model", "system_prompt", "prompt"):
            if not isinstance(generator.get(key), str) or not generator[key].strip():
                raise ValueError(
                    f"Condition '{name}' requires a non-empty params.generator.{key}."
                )
        model = params["model"]
        provider, separator, model_name = (
            model.partition("/") if isinstance(model, str) else ("", "", "")
        )
        if (
            not isinstance(model, str)
            or model.strip() != model
            or provider.lower() != "jev"
            or (separator and not model_name.strip())
        ):
            raise ValueError(f"Condition '{name}' requires a Jev model (jev/*).")
        params["max_runs"] = _positive_integer(params.get("max_runs", 2), name=name)
        return cls(name=name, prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        params = self.params
        if context.agent_state is not None:
            run_key = "jev_bag_of_decisions_judge_runs"
            context.agent_state[run_key] = context.agent_state.get(run_key, 0) + 1
            judge_runs = context.agent_state[run_key]
        else:
            judge_runs = context.num_loops

        evaluations = _run_jev_bag_of_decisions_judge(
            query=context.query,
            corpus=context.corpus,
            lookup=context.lookup,
            ranked_doc_ids=ranked_doc_ids(context.response),
            model=str(params["model"]),
            generator=params["generator"],
            positive_probability_threshold=params["positive_probability_threshold"],
            negative_probability_threshold=params["negative_probability_threshold"],
            state_format=params["state_format"],
            agent_state=context.agent_state,
            logger=context.logger,
        )
        if _jev_bag_of_decisions_judge_is_passing(evaluations):
            return ConditionResult.success()
        if judge_runs >= int(params.get("max_runs", 2)):
            return ConditionResult.success()

        feedback = _jev_bag_of_decisions_judge_feedback(evaluations)
        return self.feedback(
            f"{self.prompt}\n\nJev bag-of-decisions evaluations:\n\n"
            f"{feedback}\n\nSystem reminder: return DOC IDs ranked best to worst."
        )
