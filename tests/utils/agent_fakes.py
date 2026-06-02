from __future__ import annotations

from cheat_at_search.codegen.models import Edit
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


class FakeCodegenAgent:
    patch_edit: Edit | None = None

    def __init__(self, tools, model, response_model, reasoning_level):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level

    def chat(self, inputs=None, agent_state=None, return_usage=False, logger=None):
        if self.patch_edit is not None:
            for tool in self.tools:
                if getattr(tool, "__name__", None) == "commit_patch":
                    tool(self.patch_edit)
                    break
        if self.response_model is not None:
            try:
                output = self.response_model(
                    message="Done",
                    short_name="patch",
                    summary="Applied patch",
                )
            except Exception:
                output = None
        else:
            output = None
        resp = type("Resp", (), {"output_parsed": output})
        return resp, inputs, 0

    def loop(self, inputs=None, agent_state=None, return_usage=False, logger=None):
        resp, _, usage = self.chat(
            inputs=inputs,
            agent_state=agent_state,
            return_usage=return_usage,
            logger=logger,
        )
        if return_usage:
            return resp.output_parsed, usage
        return resp.output_parsed
