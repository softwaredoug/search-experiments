from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Protocol


ConditionKind = Literal["stop", "validator"]


@dataclass
class ConditionContext:
    """Values available while evaluating one stop condition or validator."""

    num_loops: int
    tool_calls: int
    response: Any
    query: str = ""
    corpus: Any = None
    lookup: dict | None = None
    judgments: Any = None
    agent_state: dict | None = None
    logger: Any = None
    images: bool = False


@dataclass(frozen=True)
class ConditionResult:
    """Whether a condition is satisfied, and feedback when it is not.

    For validators, ``satisfied`` means the result passed validation. For
    stoppers, it means the stop condition has been reached. The agent loop
    handles those two meanings at its boundary.
    """

    satisfied: bool
    feedback: str | None = None

    @classmethod
    def success(cls) -> ConditionResult:
        return cls(satisfied=True)

    @classmethod
    def unsatisfied(cls, feedback: str) -> ConditionResult:
        return cls(satisfied=False, feedback=feedback)


class Condition(Protocol):
    name: str
    prompt: str
    params: dict[str, Any]
    kind: ConditionKind

    def evaluate(self, context: ConditionContext) -> ConditionResult: ...

    def __getitem__(self, key: str) -> Any: ...


@dataclass
class BaseCondition(ABC):
    name: str
    prompt: str
    params: dict[str, Any]
    kind: ConditionKind

    @classmethod
    @abstractmethod
    def from_config(
        cls,
        *,
        prompt: str,
        params: dict[str, Any],
        kind: ConditionKind,
    ) -> BaseCondition: ...

    @abstractmethod
    def evaluate(self, context: ConditionContext) -> ConditionResult: ...

    def __getitem__(self, key: str) -> Any:
        """Keep the former normalized-dict access pattern available."""
        if key in {"name", "prompt", "params"}:
            return getattr(self, key)
        raise KeyError(key)

    def feedback(self, text: str | None = None) -> ConditionResult:
        return ConditionResult.unsatisfied(text if text is not None else self.prompt)


def ranked_doc_ids(response: Any) -> list[str]:
    parsed = getattr(response, "output_parsed", None) if response else None
    ranked = getattr(parsed, "ranked_results", None) if parsed else None
    return list(ranked or [])
