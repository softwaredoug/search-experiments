from __future__ import annotations

from contextlib import contextmanager
import sys
import traceback
from threading import Event, Thread, get_ident
from time import monotonic
from typing import Any, Iterator


QUERY_HEARTBEAT_INTERVAL_SECONDS = 30.0
QUERY_STACK_DUMP_INTERVAL_SECONDS = 120.0


def trace_event(logger, event: str, **fields: Any) -> None:
    if logger is not None:
        logger.info("%s %s", event, fields)


def set_trace_stage(
    agent_state: dict | None,
    phase: str,
    **fields: Any,
) -> None:
    if agent_state is None:
        return
    agent_state["_trace_stage"] = {
        "phase": phase,
        "started_at": monotonic(),
        **fields,
    }


@contextmanager
def query_heartbeat(
    *,
    query: str,
    agent_state: dict,
    logger,
    interval_seconds: float = QUERY_HEARTBEAT_INTERVAL_SECONDS,
    stack_dump_interval_seconds: float = QUERY_STACK_DUMP_INTERVAL_SECONDS,
) -> Iterator[None]:
    """Periodically log the query's current phase while a worker is busy."""
    query_started = monotonic()
    worker_thread_id = get_ident()
    stop_event = Event()
    last_stack_dump = [query_started]

    def _log_heartbeats() -> None:
        while not stop_event.wait(interval_seconds):
            now = monotonic()
            stage = dict(agent_state.get("_trace_stage") or {})
            stage_started = stage.pop("started_at", query_started)
            phase = stage.pop("phase", "unknown")
            trace_event(
                logger,
                "agentic_query_heartbeat",
                query=query,
                elapsed_seconds=round(now - query_started, 2),
                phase=phase,
                phase_elapsed_seconds=round(now - stage_started, 2),
                **stage,
            )
            if now - last_stack_dump[0] >= stack_dump_interval_seconds:
                frame = sys._current_frames().get(worker_thread_id)
                stack = (
                    "".join(traceback.format_stack(frame))
                    if frame is not None
                    else "Worker thread stack unavailable."
                )
                trace_event(
                    logger,
                    "agentic_query_stack",
                    query=query,
                    elapsed_seconds=round(now - query_started, 2),
                    phase=phase,
                    stack=stack,
                )
                last_stack_dump[0] = now

    heartbeat_thread = Thread(
        target=_log_heartbeats,
        name="agentic-query-heartbeat",
        daemon=True,
    )
    heartbeat_thread.start()
    try:
        yield
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)
