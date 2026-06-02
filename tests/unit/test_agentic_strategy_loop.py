import logging
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import exps.agentic.agent as agent_mod
import exps.agentic.strategy as agentic_mod
from exps.agentic import conditions as conditions_mod


class _FakeOpenAIAgent:
    last_instance = None

    def __init__(self, tools, model, response_model, reasoning_level):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level
        self.calls = 0
        _FakeOpenAIAgent.last_instance = self

    def chat(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        if agent_state is not None:
            agent_state["num_tool_calls"] = agent_state.get("num_tool_calls", 0) + 1
        inputs.append({"type": "function_call_output", "output": {"ok": True}})
        result = agent_mod.SearchResults(ranked_results=["101", "202", "303"])
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = inputs
        self.last_agent_state = agent_state
        return resp, inputs, 0


class _FakeJudgeOpenAIAgent(_FakeOpenAIAgent):
    def chat(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        result = conditions_mod.LLMJudgeResponse(
            graded_results=[
                conditions_mod.GradedSearchResult(
                    emoji="😞",
                    title="Red Shoes",
                    doc_id="101",
                ),
                conditions_mod.GradedSearchResult(
                    emoji="😃",
                    title="Ship Wheel",
                    doc_id="202",
                ),
            ]
        )
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = inputs
        self.last_agent_state = agent_state
        return resp, inputs, 0


class _FakeOpenAIAgentWithResults(_FakeOpenAIAgent):
    def chat(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        if agent_state is not None:
            agent_state["num_tool_calls"] = agent_state.get("num_tool_calls", 0) + 1
        inputs.append({"type": "function_call_output", "output": {"ok": True}})
        if self.calls == 1:
            result = agent_mod.SearchResults(ranked_results=["101", "202", "303"])
        else:
            result = agent_mod.SearchResults(ranked_results=["101", "202", "303", "404"])
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = inputs
        self.last_agent_state = agent_state
        return resp, inputs, 0


_MAIN_AGENT: dict[str, _FakeOpenAIAgent | None] = {"instance": None}


def _agent_factory(*args, **kwargs):
    response_model = kwargs.get("response_model")
    if response_model is conditions_mod.LLMJudgeResponse:
        return _FakeJudgeOpenAIAgent(*args, **kwargs)
    agent = _FakeOpenAIAgent(*args, **kwargs)
    _MAIN_AGENT["instance"] = agent
    return agent


def _build_fake_agent(*args, **kwargs):
    return _FakeOpenAIAgent(*args, **kwargs)


def _build_fake_agent_with_results(*args, **kwargs):
    return _FakeOpenAIAgentWithResults(*args, **kwargs)


def _sample_corpus():
    return pd.DataFrame(
        {
            "doc_id": [101, 202, 303],
            "title": ["Alpha", "Beta", "Gamma"],
            "description": ["A", "B", "C"],
        }
    )


@patch.object(agent_mod, "build_openai_agent", _build_fake_agent)
@patch.object(agent_mod, "build_search_tools", lambda *args, **kwargs: [])
def test_agentic_stop_iterations_prompt_appends(tmp_path):
    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        stop=[{"iterations": {"prompt": "Try again", "params": {"iterations": 2}}}],
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


@patch.object(agent_mod, "build_openai_agent", _build_fake_agent)
@patch.object(agent_mod, "build_search_tools", lambda *args, **kwargs: [])
def test_agentic_stop_tool_calls(tmp_path):
    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        stop=[{"tool_calls": {"prompt": "Again", "params": {"num_calls": 2}}}],
        trace_path=tmp_path,
    )
    strategy.search("query", k=2)

    agent_state = _FakeOpenAIAgent.last_instance.last_agent_state
    assert agent_state["num_tool_calls"] == 2


@patch.object(agent_mod, "build_openai_agent", _build_fake_agent_with_results)
@patch.object(agent_mod, "build_search_tools", lambda *args, **kwargs: [])
def test_agentic_validator_runs_before_stopper(tmp_path):
    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        stop=[{"iterations": {"prompt": "Stop", "params": {"iterations": 2}}}],
        validators=[{"num_results": {"prompt": "Need more results", "params": {"min_results": 4}}}],
        trace_path=tmp_path,
    )
    strategy.search("query", k=2)

    inputs = _FakeOpenAIAgentWithResults.last_instance.last_inputs
    validator_prompt_count = sum(
        1
        for item in inputs
        if isinstance(item, dict)
        and item.get("role") == "user"
        and item.get("content") == "Need more results"
    )
    stop_prompt_count = sum(
        1
        for item in inputs
        if isinstance(item, dict)
        and item.get("role") == "user"
        and item.get("content") == "Stop"
    )
    assert validator_prompt_count == 1
    assert stop_prompt_count == 0


@patch.object(agent_mod, "build_openai_agent", _agent_factory)
@patch.object(conditions_mod, "OpenAIAgent", _agent_factory)
@patch.object(agent_mod, "build_search_tools", lambda *args, **kwargs: [])
def test_llm_judge_validator_appends_prompt(tmp_path):
    _MAIN_AGENT["instance"] = None
    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        validators=[
            {
                "llm_judge_relevance": {
                    "prompt": "Please return more relevant results.",
                    "params": {
                        "model": "gpt-5-mini",
                        "reasoning": "medium",
                        "judge_prompt": "Query: {query}\nResults:\n{results}",
                    },
                }
            }
        ],
        max_loops=2,
        trace_path=tmp_path,
    )
    strategy.search("query", k=2)

    inputs = _MAIN_AGENT["instance"].last_inputs
    prompt_count = sum(
        1
        for item in inputs
        if isinstance(item, dict)
        and item.get("role") == "user"
        and "LLM evaluations" in str(item.get("content"))
    )
    assert prompt_count == 1


@patch.object(agent_mod, "build_openai_agent", _build_fake_agent)
@patch.object(agent_mod, "build_search_tools", lambda *args, **kwargs: [])
def test_agentic_max_loops_stops(tmp_path):
    strategy = agentic_mod.AgenticSearchStrategy(
        _sample_corpus(),
        workers=1,
        model="gpt-5-mini",
        search_tools=[],
        validators=[{"num_results": {"prompt": "Need more", "params": {"min_results": 10}}}],
        max_loops=2,
        trace_path=tmp_path,
    )
    strategy.search("query", k=2)

    assert _FakeOpenAIAgent.last_instance.calls == 2


@patch.object(agent_mod, "build_openai_agent", _build_fake_agent)
@patch.object(agent_mod, "build_search_tools", lambda *args, **kwargs: [])
def test_trace_log_records_outputs(tmp_path, caplog):
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
