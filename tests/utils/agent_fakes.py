from __future__ import annotations

from exps.agentic.agent import SearchResults


class FakeOpenAIAgent:
    calls = 0
    doc_ids: list[str] = []
    categories: list[str] = []

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
        output = self._build_output()
        resp = type("Resp", (), {"output_parsed": output})
        return resp, inputs, 0

    def _build_output(self):
        model = self.response_model
        if model is None:
            return None
        fields = getattr(model, "model_fields", None)
        if fields and "categories" in fields:
            return model(categories=list(FakeOpenAIAgent.categories))
        if fields and "ranked_results" in fields:
            return model(ranked_results=list(FakeOpenAIAgent.doc_ids))
        try:
            return model()
        except TypeError:
            return SearchResults(ranked_results=list(FakeOpenAIAgent.doc_ids))
