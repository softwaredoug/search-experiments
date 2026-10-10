from __future__ import annotations

from threading import Event
from types import SimpleNamespace

from exps.agentic.agent import (
    TracingOpenAIAgent,
    _instrument_search_tool,
    build_openai_agent,
)
from exps.agentic.conditions import _TracingRetryPolicy
from exps.agentic.tracing import query_heartbeat, set_trace_stage
from typesafe_sdk import TypeSafeAPITimeoutError


class _CaptureLogger:
    def __init__(self):
        self.events = []

    def info(self, message, *args):
        self.events.append(message % args if args else message)

    def warning(self, message, *args):
        self.events.append(message % args if args else message)


def _response():
    return SimpleNamespace(
        output=[SimpleNamespace(type="function_call", name="search_bm25")],
        _request_id="resp_123",
    )


def _tracing_agent(create_response):
    agent = object.__new__(TracingOpenAIAgent)
    agent.response_model = None
    agent.model = "gpt-5-mini"
    agent._trace_request_index = 0
    agent._active_agent_state = {}
    agent.openai = SimpleNamespace(
        responses=SimpleNamespace(create=create_response),
    )
    return agent


def test_openai_request_events_include_attempt_duration_and_request_id(monkeypatch):
    logger = _CaptureLogger()
    attempts = []

    def create(**_kwargs):
        attempts.append(True)
        if len(attempts) == 1:
            raise TimeoutError("request timed out")
        return _response()

    monkeypatch.setattr(
        "cheat_at_search.agent.openai_agent.sleep",
        lambda _seconds: None,
    )
    agent = _tracing_agent(create)
    response = agent._call_responses_with_retry(
        inputs=[],
        tools=[],
        reasoning={},
        active_logger=logger,
    )

    assert response._request_id == "resp_123"
    assert len(attempts) == 2
    assert sum("agentic_openai_request_start" in event for event in logger.events) == 1
    assert any("agentic_openai_request_retry" in event for event in logger.events)
    assert any("request_id': 'resp_123'" in event for event in logger.events)
    assert any("attempt': 2" in event for event in logger.events)


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


def test_search_tool_wrapper_logs_start_completion_and_elapsed_time():
    logger = _CaptureLogger()
    state = {"trace_logger": logger}
    calls = []

    def call_from_tool(args_model, *, agent_state):
        calls.append((args_model, agent_state))
        return {"ok": True}, '{"ok": true}'

    timed_tool = _instrument_search_tool("search_bm25", call_from_tool)
    args = SimpleNamespace(keywords="dresser")
    result = timed_tool(args, agent_state=state)

    assert result == ({"ok": True}, '{"ok": true}')
    assert calls == [(args, state)]
    assert state["_trace_tool_calls"] == 1
    assert any("agentic_tool_start" in event for event in logger.events)
    completion = next(
        event for event in logger.events if "agentic_tool_complete" in event
    )
    assert "search_bm25" in completion
    assert "elapsed_seconds" in completion


def test_tracing_openai_agent_preserves_tool_schema_and_instruments_adapter(
    monkeypatch,
):
    monkeypatch.setattr(
        "cheat_at_search.agent.openai_agent.OpenAI",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(
        "cheat_at_search.agent.openai_agent.key_for_provider",
        lambda _provider: "test-key",
    )

    def search_bm25(*, keywords: str, agent_state: dict | None = None) -> dict:
        """Search products with BM25."""
        return {"query": keywords}

    agent = build_openai_agent(
        tools=[search_bm25],
        model="openai/gpt-5-mini",
        response_model=None,
        reasoning_level="low",
    )
    logger = _CaptureLogger()
    state = {"trace_logger": logger}
    args_model, tool_spec, tool_call = agent.search_tools["search_bm25"]

    assert tool_spec["name"] == "search_bm25"
    result, json_result = tool_call(
        args_model(keywords="dresser"),
        agent_state=state,
    )

    assert result == {"query": "dresser"}
    assert json_result == '{"query":"dresser"}'
    assert any("agentic_tool_start" in event for event in logger.events)
    assert any("agentic_tool_complete" in event for event in logger.events)


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
