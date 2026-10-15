from __future__ import annotations

from threading import Event

from exps.agentic.conditions.jev_bag_of_decisions import _TracingRetryPolicy
from exps.agentic.tracing import query_heartbeat, set_trace_stage
from typesafe_sdk import TypeSafeAPITimeoutError


class _CaptureLogger:
    def __init__(self):
        self.events = []

    def info(self, message, *args):
        self.events.append(message % args if args else message)

    def warning(self, message, *args):
        self.events.append(message % args if args else message)


def test_jev_retry_policy_logs_retryable_failures_and_exhaustion():
    logger = _CaptureLogger()
    policy = _TracingRetryPolicy(
        logger=logger,
        context={
            "doc_index": 2,
            "total_docs": 7,
            "doc_id": "202",
        },
    )

    assert policy._retryable(TypeSafeAPITimeoutError(10)) is True
    assert policy._retryable(TypeSafeAPITimeoutError(10)) is True
    assert policy._retryable(TypeSafeAPITimeoutError(10)) is True

    checks = [
        event
        for event in logger.events
        if "agentic_jev_request_retry_check" in event
    ]
    assert len(checks) == 3
    assert all("TypeSafeAPITimeoutError" in event for event in checks)
    assert "retrying': False" in checks[-1]


def test_query_heartbeat_reports_active_stage():
    logger = _CaptureLogger()
    state = {}
    set_trace_stage(state, "jev_document_evaluation", doc_index=3, total_docs=12)

    with query_heartbeat(
        query="parsons chairs",
        agent_state=state,
        logger=logger,
        interval_seconds=0.01,
        stack_dump_interval_seconds=0.01,
    ):
        Event().wait(0.035)

    heartbeat = next(
        event for event in logger.events if "agentic_query_heartbeat" in event
    )
    assert "parsons chairs" in heartbeat
    assert "jev_document_evaluation" in heartbeat
    assert "doc_index': 3" in heartbeat
    assert "phase_elapsed_seconds" in heartbeat
    stack_dump = next(
        event for event in logger.events if "agentic_query_stack" in event
    )
    assert "test_query_heartbeat_reports_active_stage" in stack_dump
