from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass
class ModelCallCounter:
    calls: int = 0


_model_call_counter: ContextVar[ModelCallCounter | None] = ContextVar(
    "model_call_counter", default=None
)


@contextmanager
def track_model_calls() -> Iterator[ModelCallCounter]:
    counter = ModelCallCounter()
    token = _model_call_counter.set(counter)
    try:
        yield counter
    finally:
        _model_call_counter.reset(token)


def record_model_call() -> None:
    counter = _model_call_counter.get()
    if counter is not None:
        counter.calls += 1


def resources_from_agent_state(
    agent_state: dict,
    *,
    tool_calls: int | None = None,
) -> dict[str, int | float]:
    resources = {
        key: value
        for key, value in agent_state.items()
        if key.startswith("resource.") and key != "resource.num_tool_calls"
    }
    if tool_calls is None:
        tool_calls = agent_state.get("resource.num_tool_calls")
    if tool_calls is not None:
        resources["resource.tool_calls"] = tool_calls
    return resources
