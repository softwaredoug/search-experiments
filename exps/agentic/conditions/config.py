from __future__ import annotations

from typing import Any, Mapping

from exps.agentic.conditions.base import BaseCondition, Condition, ConditionKind
from exps.agentic.conditions.jev_bag_of_decisions import JevBagOfDecisionsJudge
from exps.agentic.conditions.jev_judge import JevJudgeRelevance
from exps.agentic.conditions.llm_judge import LLMJudgeRelevance
from exps.agentic.conditions.oracle import OracleValidator
from exps.agentic.conditions.simple import (
    IterationsCondition,
    NumResultsCondition,
    ToolCallsCondition,
)


_FACTORIES: dict[str, type[BaseCondition]] = {
    "iterations": IterationsCondition,
    "tool_calls": ToolCallsCondition,
    "num_results": NumResultsCondition,
    "llm_judge_relevance": LLMJudgeRelevance,
    "jev_judge_relevance": JevJudgeRelevance,
    "jev_bag_of_decisions_judge": JevBagOfDecisionsJudge,
    "oracle": OracleValidator,
}


def _parse_entry(entry: Any) -> tuple[str, dict[str, Any]]:
    if isinstance(entry, str):
        return entry, {}
    if not (isinstance(entry, dict) and len(entry) == 1):
        raise ValueError("Condition entries must be single-key mappings.")
    (name, raw_params), = entry.items()
    if not isinstance(raw_params, dict):
        raise ValueError("Condition params must be a mapping.")
    return name, dict(raw_params)


def _require_prompt(prompt: Any, *, kind: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{kind} condition requires a non-empty prompt.")
    return prompt


def _condition_params(
    name: str,
    outer_params: dict[str, Any],
    *,
    kind: ConditionKind,
) -> tuple[str, dict[str, Any]]:
    prompt = outer_params.get("prompt")
    if not prompt and name == "oracle":
        prompt = "Please return more relevant results."
    prompt = _require_prompt(prompt, kind=kind)
    outer_params.pop("prompt", None)
    params = outer_params.get("params")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError(f"{kind} condition '{name}' requires params mapping.")
    return prompt, dict(params)


def normalize_conditions(
    condition_config: list | None,
    *,
    kind: ConditionKind,
) -> list[Condition]:
    """Parse each YAML entry and delegate its schema to its condition type."""
    if not condition_config:
        return []
    conditions = []
    for entry in condition_config:
        name, outer_params = _parse_entry(entry)
        prompt, params = _condition_params(name, outer_params, kind=kind)
        factory = _FACTORIES.get(name)
        if factory is None:
            raise ValueError(f"Unknown {kind} condition: {name}")
        conditions.append(
            factory.from_config(prompt=prompt, params=params, kind=kind)
        )
    return conditions


def condition_from_mapping(
    condition: Mapping[str, Any], *, kind: ConditionKind
) -> Condition:
    """Adapt an already-normalized mapping to the typed runtime contract."""
    name = condition["name"]
    factory = _FACTORIES.get(name)
    if factory is None:
        raise ValueError(f"Unknown {kind} condition: {name}")
    return factory.from_config(
        prompt=condition["prompt"], params=dict(condition["params"]), kind=kind
    )
