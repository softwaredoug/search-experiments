from __future__ import annotations

from dataclasses import dataclass
from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
)


def _num_results_from_response(response) -> int:
    if response is None:
        return 0
    parsed = getattr(response, "output_parsed", None)
    ranked = getattr(parsed, "ranked_results", None) if parsed is not None else None
    return len(ranked or [])


@dataclass
class IterationsCondition(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> IterationsCondition:
        if "iterations" not in params:
            raise ValueError("Condition 'iterations' requires params.iterations.")
        return cls(name="iterations", prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        if context.num_loops >= int(self.params["iterations"]):
            return ConditionResult.success()
        return self.feedback()


@dataclass
class ToolCallsCondition(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> ToolCallsCondition:
        if "num_calls" not in params:
            raise ValueError("Condition 'tool_calls' requires params.num_calls.")
        return cls(name="tool_calls", prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        if context.tool_calls >= int(self.params["num_calls"]):
            return ConditionResult.success()
        return self.feedback()


@dataclass
class NumResultsCondition(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> NumResultsCondition:
        if "min_results" not in params:
            raise ValueError("Condition 'num_results' requires params.min_results.")
        return cls(name="num_results", prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        if _num_results_from_response(context.response) >= int(self.params["min_results"]):
            return ConditionResult.success()
        return self.feedback()
