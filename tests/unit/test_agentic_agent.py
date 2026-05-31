import copy
from unittest.mock import patch

import pandas as pd
import pytest

import exps.agentic.agent as agent_mod
from exps.agentic.agent import trace_logger


class _FakeOpenAIAgent:
    last_instance = None
    instances: list["_FakeOpenAIAgent"] = []

    def __init__(self, tools, model, response_model, reasoning_level):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level
        self.calls = 0
        _FakeOpenAIAgent.last_instance = self
        _FakeOpenAIAgent.instances.append(self)

    def chat(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        inputs.append({"type": "function_call_output", "output": {"ok": True}})
        result = agent_mod.SearchResults(ranked_results=["101", "202", "303"])
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = copy.deepcopy(inputs)
        return resp, inputs, 0


class _FakeOpenAIAgentNoTools(_FakeOpenAIAgent):
    def chat(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        result = agent_mod.SearchResults(ranked_results=["101", "202", "303"])
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = copy.deepcopy(inputs)
        return resp, inputs, 0


@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.OpenAIAgent", new=_FakeOpenAIAgent)
def test_agent_runs_single_step(tmp_path):
    _FakeOpenAIAgent.instances = []

    corpus = pd.DataFrame({"doc_id": [101, 202, 303]})
    agent = agent_mod.Agent(
        corpus=corpus,
        model="gpt-5-mini",
        reasoning="low",
        system_prompt="system",
        search_tools=["bm25"],
        plan=None,
    )
    with trace_logger(tmp_path) as (logger, trace_path):
        result = agent.run(
            query="query",
            trace_dir=tmp_path,
            logger=logger,
            trace_path=trace_path,
            k=2,
        )

    assert result.output == ["101", "202"]
    assert result.num_tool_calls == 1


@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.OpenAIAgent", new=_FakeOpenAIAgent)
def test_agent_plan_switches_system_prompt(tmp_path):
    _FakeOpenAIAgent.instances = []

    agent = agent_mod.Agent(
        corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
        model="gpt-5-mini",
        reasoning="low",
        system_prompt="default",
        agents={
            "planning": {"system_prompt": "planning prompt", "search_tools": ["bm25"]},
            "search": {"system_prompt": "search prompt", "search_tools": ["bm25"]},
        },
        plan=[
            {"planning": "plan for {query}"},
            {"search": "find results for {query}"},
        ],
    )

    with trace_logger(tmp_path) as (logger, trace_path):
        result = agent.run(
            query="shoes",
            trace_dir=tmp_path,
            logger=logger,
            trace_path=trace_path,
            k=2,
        )

    assert len(_FakeOpenAIAgent.instances) == 2
    assert _FakeOpenAIAgent.instances[0].last_inputs[0]["content"] == "planning prompt"
    assert _FakeOpenAIAgent.instances[1].last_inputs[0]["content"] == "search prompt"
    assert result.num_tool_calls == 2


@patch("exps.agentic.agent.evaluate_stopper", new=lambda *args, **kwargs: True)
@patch("exps.agentic.agent.evaluate_validator", new=lambda *args, **kwargs: True)
@patch("exps.agentic.agent.normalize_conditions", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.OpenAIAgent", new=_FakeOpenAIAgentNoTools)
def test_agent_empty_tools_no_tool_calls(tmp_path):
    _FakeOpenAIAgent.instances = []
    agent = agent_mod.Agent(
        corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
        search_tools=[],
    )

    with trace_logger(tmp_path) as (logger, trace_path):
        result = agent.run(
            query="query",
            trace_dir=tmp_path,
            logger=logger,
            trace_path=trace_path,
            k=2,
        )

    assert result.num_tool_calls == 0
    assert result.output == ["101", "202"]


@patch("exps.agentic.agent.evaluate_stopper")
@patch("exps.agentic.agent.evaluate_validator")
@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.OpenAIAgent", new=_FakeOpenAIAgent)
def test_agent_validators_then_stop(
    mock_validator,
    mock_stopper,
    tmp_path,
):
    _FakeOpenAIAgent.instances = []
    mock_validator.side_effect = ["Fix results", True, True]
    mock_stopper.side_effect = ["Try again", True]

    agent = agent_mod.Agent(
        corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
        search_tools=["bm25"],
        validators=[{"num_results": {"prompt": "Fix results", "params": {"min_results": 3}}}],
        stop=[{"iterations": {"prompt": "Try again", "params": {"iterations": 2}}}],
        max_loops=3,
    )

    with trace_logger(tmp_path) as (logger, trace_path):
        result = agent.run(
            query="query",
            trace_dir=tmp_path,
            logger=logger,
            trace_path=trace_path,
            k=2,
        )

    assert result.num_tool_calls == 3
    assert mock_validator.call_count == 2
    assert mock_stopper.call_count == 1


@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.OpenAIAgent", new=_FakeOpenAIAgent)
def test_agent_unknown_plan_agent_raises():
    with pytest.raises(ValueError, match="Unknown plan agent"):
        agent_mod.Agent(
            corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
            agents={"planning": {"system_prompt": "planning", "search_tools": ["bm25"]}},
            plan=[{"search": "search for {query}"}],
        )


@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.OpenAIAgent", new=_FakeOpenAIAgent)
def test_agent_plan_formats_user_prompt(tmp_path):
    _FakeOpenAIAgent.instances = []
    agent = agent_mod.Agent(
        corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
        agents={"planning": {"system_prompt": "planning", "search_tools": ["bm25"]}},
        plan=[{"planning": "plan for {query}"}],
    )

    with trace_logger(tmp_path) as (logger, trace_path):
        agent.run(
            query="blue chair",
            trace_dir=tmp_path,
            logger=logger,
            trace_path=trace_path,
            k=2,
        )

    user_prompt = _FakeOpenAIAgent.instances[0].last_inputs[1]["content"]
    assert user_prompt == "plan for blue chair"
