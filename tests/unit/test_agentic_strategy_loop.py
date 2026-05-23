import logging
from pathlib import Path

import pandas as pd

import exps.agentic.strategy as agentic_mod


class _FakeOpenAIAgent:
    last_instance = None

    def __init__(self, tools, model, response_model, reasoning_level):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level
        self.calls = 0
        _FakeOpenAIAgent.last_instance = self

    def chat(self, inputs=None, agent_state=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        if agent_state is not None:
            agent_state["num_tool_calls"] = agent_state.get("num_tool_calls", 0) + 1
        inputs.append({"type": "function_call_output", "output": {"ok": True}})
        result = agentic_mod.SearchResultsIds(
            results_summary="ok",
            next_plan="next",
            ranked_results=["101", "202", "303"],
        )
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = inputs
        self.last_agent_state = agent_state
        return resp, inputs, 0


def _sample_corpus():
    return pd.DataFrame(
        {
            "doc_id": [101, 202, 303],
            "title": ["Alpha", "Beta", "Gamma"],
            "description": ["A", "B", "C"],
        }
    )


def test_agentic_stop_iterations_reprompt_appends(monkeypatch, tmp_path):
    monkeypatch.setattr(agentic_mod, "OpenAIAgent", _FakeOpenAIAgent)
    monkeypatch.setattr(agentic_mod, "build_search_tools", lambda *args, **kwargs: [])

    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        stop=[{"iterations": 2}],
        reprompt="Try again",
        trace_path=tmp_path,
    )
    strategy.search("query", k=2)

    inputs = _FakeOpenAIAgent.last_instance.last_inputs
    reprompt_count = sum(
        1
        for item in inputs
        if isinstance(item, dict)
        and item.get("role") == "user"
        and item.get("content") == "Try again"
    )
    assert reprompt_count == 1
    assert _FakeOpenAIAgent.last_instance.calls == 2


def test_agentic_stop_tool_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(agentic_mod, "OpenAIAgent", _FakeOpenAIAgent)
    monkeypatch.setattr(agentic_mod, "build_search_tools", lambda *args, **kwargs: [])

    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        stop=[{"tool_calls": 2}],
        reprompt="Again",
        trace_path=tmp_path,
    )
    strategy.search("query", k=2)

    agent_state = _FakeOpenAIAgent.last_instance.last_agent_state
    assert agent_state["num_tool_calls"] == 2


def test_trace_log_records_outputs(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(agentic_mod, "OpenAIAgent", _FakeOpenAIAgent)
    monkeypatch.setattr(agentic_mod, "build_search_tools", lambda *args, **kwargs: [])

    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        trace_path=tmp_path,
    )
    with caplog.at_level(logging.INFO):
        strategy.search("query", k=2)

    trace_path = strategy.traces["query"]
    content = Path(trace_path).read_text(encoding="utf-8")
    assert "agentic_output" in content
