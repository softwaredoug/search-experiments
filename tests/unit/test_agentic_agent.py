import copy

import pandas as pd

import exps.agentic.agent as agent_mod


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


def test_agent_runs_single_step(tmp_path, monkeypatch):
    _FakeOpenAIAgent.instances = []
    monkeypatch.setattr(agent_mod, "OpenAIAgent", _FakeOpenAIAgent)
    monkeypatch.setattr(agent_mod, "build_search_tools", lambda *args, **kwargs: [])

    corpus = pd.DataFrame({"doc_id": [101, 202, 303]})
    agent = agent_mod.Agent(
        corpus=corpus,
        model="gpt-5-mini",
        reasoning="low",
        system_prompt="system",
        search_tools=["bm25"],
        plan=None,
    )
    result = agent.run(query="query", trace_dir=tmp_path, k=2)

    assert result.ranked_results == ["101", "202"]
    assert result.num_tool_calls == 1


def test_agent_plan_switches_system_prompt(tmp_path, monkeypatch):
    _FakeOpenAIAgent.instances = []
    monkeypatch.setattr(agent_mod, "OpenAIAgent", _FakeOpenAIAgent)
    monkeypatch.setattr(agent_mod, "build_search_tools", lambda *args, **kwargs: [])

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

    result = agent.run(query="shoes", trace_dir=tmp_path, k=2)

    assert len(_FakeOpenAIAgent.instances) == 2
    assert _FakeOpenAIAgent.instances[0].last_inputs[0]["content"] == "planning prompt"
    assert _FakeOpenAIAgent.instances[1].last_inputs[0]["content"] == "search prompt"
    assert result.num_tool_calls == 2
