from __future__ import annotations

from exps.agentic.agent import SearchResults


class FakeOpenAIAgent:
    calls = 0
    doc_ids: list[str] = []

    def __init__(self, tools, model, response_model, reasoning_level):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level

    def chat(self, *, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        FakeOpenAIAgent.calls += 1
        inputs.append({"type": "function_call_output", "output": {"ok": True}})
        result = SearchResults(ranked_results=FakeOpenAIAgent.doc_ids)
        resp = type("Resp", (), {"output_parsed": result})
        return resp, inputs, 0
