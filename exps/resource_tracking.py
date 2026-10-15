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
