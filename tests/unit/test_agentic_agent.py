import copy
from unittest.mock import patch

import pandas as pd
import pytest

from cheat_at_search.agent.sid_tools import build_sid2_tool_config

import exps.agentic.agent as agent_mod
from exps.agentic.agent import trace_logger


class _FakeOpenAIAgent:
    last_instance = None
    instances: list["_FakeOpenAIAgent"] = []

    def __init__(self, tools, model, response_model, reasoning_level, images=False):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level
        self.images = images
        self.calls = 0
        _FakeOpenAIAgent.last_instance = self
        _FakeOpenAIAgent.instances.append(self)

    def loop(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        inputs.append({"type": "function_call_output", "output": {"ok": True}})
        result = agent_mod.SearchResults(ranked_results=["101", "202", "303"])
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = copy.deepcopy(inputs)
        return resp, inputs


class _FakeOpenAIAgentNoTools(_FakeOpenAIAgent):
    def loop(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        result = agent_mod.SearchResults(ranked_results=["101", "202", "303"])
        resp = type("Resp", (), {"output_parsed": result})
        self.last_inputs = copy.deepcopy(inputs)
        return resp, inputs


class _FakeSIDAgent:
    instances: list["_FakeSIDAgent"] = []

    def __init__(self, tools, max_turns, model):
        self.tools = tools
        self.max_turns = max_turns
        self.model = model
        self.logger = None
        type(self).instances.append(self)

    def loop(self, inputs=None, agent_state=None, logger=None):
        self.logger = logger
        inputs.append(
            {"role": "tool", "tool_call_id": "call_1", "content": "search results"}
        )
        agent_state["trace_logger"] = logger
        result = agent_mod.SearchResults(ranked_results=["101", "202", "303"])
        resp = type("Resp", (), {"output_parsed": result})
        return resp, inputs


def _build_fake_agent(*args, **kwargs):
    return _FakeOpenAIAgent(*args, **kwargs)


def _build_fake_agent_no_tools(*args, **kwargs):
    return _FakeOpenAIAgentNoTools(*args, **kwargs)


@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.build_openai_agent", new=_build_fake_agent)
def test_agent_runs_single_step(tmp_path):
    _FakeOpenAIAgent.instances = []

    corpus = pd.DataFrame({"doc_id": [101, 202, 303]})
    agent = agent_mod.Agent(
        corpus=corpus,
        model="gpt-5-mini",
        reasoning="low",
        images=True,
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
    assert _FakeOpenAIAgent.instances[0].images is True
    trace_text = trace_path.read_text(encoding="utf-8")
    assert "agentic_query_start" in trace_text
    assert "agentic_chat_start" in trace_text
    assert "agentic_chat_complete" in trace_text
    assert "agentic_query_complete" in trace_text


@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.build_openai_agent", new=_build_fake_agent)
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
@patch("exps.agentic.agent.build_openai_agent", new=_build_fake_agent_no_tools)
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
@patch("exps.agentic.agent.build_openai_agent", new=_build_fake_agent)
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
        validators=[{"oracle": {"prompt": "Fix results"}}],
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
@patch("exps.agentic.agent.build_openai_agent", new=_build_fake_agent)
def test_agent_unknown_plan_agent_raises():
    with pytest.raises(ValueError, match="Unknown plan agent"):
        agent_mod.Agent(
            corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
            agents={"planning": {"system_prompt": "planning", "search_tools": ["bm25"]}},
            plan=[{"search": "search for {query}"}],
        )


@patch("exps.agentic.agent.build_search_tools", new=lambda *args, **kwargs: [])
@patch("exps.agentic.agent.build_openai_agent", new=_build_fake_agent)
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


def test_agent_routes_sid1_to_legacy_agent_with_configured_read_tool(tmp_path):
    _FakeSIDAgent.instances = []

    def search_tool(query: str):
        return [{"id": 101, "title": "Chair", "description": "A chair."}]

    def read(id: str):
        return [{"id": id, "title": "Chair", "description": "A chair."}]

    with patch.object(agent_mod, "build_search_tools", return_value=[search_tool, read]), patch.object(
        agent_mod, "SIDLegacyAgent", new=_FakeSIDAgent
    ), patch.object(
        agent_mod, "SIDAgent", side_effect=AssertionError("SID-1 should use SIDLegacyAgent")
    ), patch.object(
        agent_mod,
        "build_openai_agent",
        side_effect=AssertionError("SID model should use a SID agent"),
    ):
        agent = agent_mod.Agent(
            corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
            model="sid/sid-1",
            search_tools=["bm25", "read"],
            max_loops=4,
        )
        with trace_logger(tmp_path) as (logger, trace_path):
            result = agent.run(
                query="query",
                trace_dir=tmp_path,
                logger=logger,
                trace_path=trace_path,
                k=2,
            )

    sid_agent = _FakeSIDAgent.instances[0]
    assert sid_agent.model == "sid-1"
    assert sid_agent.max_turns == 4
    assert sid_agent.logger is not None
    assert [tool.__name__ for tool in sid_agent.tools] == ["search_tool", "read"]
    assert result.output == ["101", "202"]
    assert result.num_tool_calls == 1


def test_agent_routes_sid2_and_adapts_search_documents(tmp_path):
    _FakeSIDAgent.instances = []
    original_results = [
        {"id": 101, "title": "Chair", "description": "A chair."},
    ]

    def search_tool(query: str):
        """Search documents for a query."""
        return original_results

    search_tool.aliases = ["text_search"]

    with patch.object(agent_mod, "build_search_tools", return_value=[search_tool]), patch.object(
        agent_mod, "SIDAgent", new=_FakeSIDAgent
    ), patch.object(
        agent_mod,
        "SIDLegacyAgent",
        side_effect=AssertionError("SID-2 should use SIDAgent"),
    ), patch.object(
        agent_mod,
        "build_openai_agent",
        side_effect=AssertionError("SID model should use a SID agent"),
    ):
        agent = agent_mod.Agent(
            corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
            model="sid/sid-2",
            search_tools=["bm25"],
            max_loops=4,
        )
        with trace_logger(tmp_path) as (logger, trace_path):
            result = agent.run(
                query="query",
                trace_dir=tmp_path,
                logger=logger,
                trace_path=trace_path,
                k=2,
            )

    sid_agent = _FakeSIDAgent.instances[0]
    assert sid_agent.model == "sid-2"
    assert sid_agent.max_turns == 4
    assert [tool.__name__ for tool in sid_agent.tools] == ["text_search"]
    assert sid_agent.tools[0].aliases == ["text_search"]
    adapted_results = sid_agent.tools[0]("query")
    assert adapted_results == [
        {
            "id": "101",
            "title": "Chair",
            "description": "A chair.",
            "content": "A chair.",
        },
    ]
    assert original_results == [
        {"id": 101, "title": "Chair", "description": "A chair."},
    ]
    tool_specs, _ = build_sid2_tool_config(sid_agent.tools)
    assert [spec["name"] for spec in tool_specs] == [
        "text_search",
        "read",
        "report_helpful_ids",
    ]
    assert "query" in tool_specs[0]["parameters"]["properties"]
    assert result.output == ["101", "202"]
    assert result.num_tool_calls == 1


def test_agent_rejects_unsupported_sid_model(tmp_path):
    agent = agent_mod.Agent(
        corpus=pd.DataFrame({"doc_id": [101, 202, 303]}),
        model="sid/sid-3",
        search_tools=[],
    )

    with trace_logger(tmp_path) as (logger, _):
        with pytest.raises(ValueError, match="expected 'sid-1' or 'sid-2'"):
            agent._run_plan_agent(
                agent_name="default",
                step_index=1,
                query="query",
                system_prompt="system",
                user_prompt="query",
                tools=[],
                inputs=[
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "query"},
                ],
                agent_state={},
                stops=[],
                validators=[],
                logger=logger,
            )


def test_sid2_uses_one_canonical_alias_per_search_family():
    def bm25_first(keywords: str):
        """Search the first BM25 index."""

    def bm25_second(keywords: str):
        """Search the second BM25 index."""

    bm25_first.aliases = ["text_search"]
    bm25_second.aliases = ["text_search"]
    tools = agent_mod._adapt_tools_for_sid2([bm25_first, bm25_second])

    assert [tool.__name__ for tool in tools] == ["text_search", "bm25_second"]
    assert tools[0].aliases == ["text_search"]
    assert tools[1].aliases == []
    tool_specs, _ = build_sid2_tool_config(tools)
    assert [spec["name"] for spec in tool_specs[:2]] == [
        "text_search",
        "bm25_second",
    ]
