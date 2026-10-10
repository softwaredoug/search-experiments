"""Condition configuration and runtime contracts for the agentic loop."""

from __future__ import annotations

from typing import Any

from exps.agentic.conditions.base import (
    BaseCondition,
    Condition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
)
from exps.agentic.conditions import judging
from exps.agentic.conditions.config import condition_from_mapping, normalize_conditions

_TracingRetryPolicy = judging._TracingRetryPolicy
GradedSearchResult = judging.GradedSearchResult
LLMJudgeResponse = judging.LLMJudgeResponse


def _as_condition(condition: Condition | dict[str, Any], *, kind: ConditionKind) -> Condition:
    if isinstance(condition, BaseCondition):
        return condition
    return condition_from_mapping(condition, kind=kind)


def evaluate_validator(
    condition: Condition | dict[str, Any],
    *,
    num_loops: int,
    tool_calls: int,
    resp,
    query: str,
    corpus,
    lookup: dict | None,
    judgments,
    agent_state: dict | None,
    logger,
    images: bool = False,
) -> bool | str:
    runtime_condition = _as_condition(condition, kind="validator")
    result = runtime_condition.evaluate(
        ConditionContext(
            num_loops=num_loops,
            tool_calls=tool_calls,
            response=resp,
            query=query,
            corpus=corpus,
            lookup=lookup,
            judgments=judgments,
            agent_state=agent_state,
            logger=logger,
            images=images,
        )
    )
    return True if result.satisfied else (result.feedback or runtime_condition.prompt)


def evaluate_stopper(
    condition: Condition | dict[str, Any],
    *,
    num_loops: int,
    tool_calls: int,
    resp,
) -> bool | str:
    runtime_condition = _as_condition(condition, kind="stop")
    result = runtime_condition.evaluate(
        ConditionContext(
            num_loops=num_loops,
            tool_calls=tool_calls,
            response=resp,
        )
    )
    return True if result.satisfied else (result.feedback or runtime_condition.prompt)


__all__ = [
    "Condition",
    "ConditionContext",
    "ConditionResult",
    "GradedSearchResult",
    "LLMJudgeResponse",
    "evaluate_stopper",
    "evaluate_validator",
    "normalize_conditions",
]
