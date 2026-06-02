from __future__ import annotations

from exps.agentic.agent import SearchResults


def build_agent_script(
    *,
    tool_name: str | None = None,
    params: dict | None = None,
    output: dict | None = None,
) -> list[dict]:
    steps: list[dict] = []
    if tool_name:
        steps.append({"function_call": {"name": tool_name, "params": params}})
    steps.append({"output": output})
    return steps


class FakeOpenAIAgent:
    def __init__(self, tools, model, response_model, reasoning_level):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level
        self.calls = 0
        self.chat_calls = 0
        self._script_index = 0
        self.doc_ids: list[str] = []
        self.categories: list[str] = []
        self.scripts: list[list[dict]] | None = None

    def chat(self, *, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        self.chat_calls += 1
        script = self._next_script()
        output = self._run_script(inputs, script)
        resp = type("Resp", (), {"output_parsed": output})
        return resp, inputs, 0

    def _build_output(self):
        model = self.response_model
        if model is None:
            return None
        fields = getattr(model, "model_fields", None)
        if fields and "categories" in fields:
            return model(categories=list(self.categories))
        if fields and "ranked_results" in fields:
            return model(ranked_results=list(self.doc_ids))
        try:
            return model()
        except TypeError:
            return SearchResults(ranked_results=list(self.doc_ids))

    def _next_script(self) -> list[dict]:
        if self.scripts is None:
            raise ValueError("FakeOpenAIAgent.scripts must be set for scripted tool calls.")
        if self._script_index >= len(self.scripts):
            script = self.scripts[-1]
            self._script_index += 1
            return script
        script = self.scripts[self._script_index]
        self._script_index += 1
        return script

    def _run_script(self, inputs, script):
        output = None
        for step in script or []:
            if "function_call" in step:
                call = step["function_call"] or {}
                name = call.get("name")
                params = call.get("params")
                if not name:
                    raise ValueError("Scripted function_call missing name.")
                tool_found = False
                for tool in self.tools:
                    if getattr(tool, "__name__", None) == name:
                        tool_found = True
                        if params is None:
                            tool()
                        elif isinstance(params, dict):
                            tool(**params)
                        else:
                            tool(params)
                        inputs.append({"type": "function_call_output", "output": {"ok": True}})
                        break
                if not tool_found:
                    raise ValueError(f"Tool '{name}' not found in FakeOpenAIAgent.tools")
            if "output" in step:
                output = self._normalize_output(step["output"])
        return output

    def _normalize_output(self, output):
        if output is None:
            return self._build_output()
        if self.response_model is None:
            return output
        if isinstance(output, self.response_model):
            return output
        if isinstance(output, dict):
            try:
                return self.response_model(**output)
            except Exception:
                return output
        return output


class FakeCodegenAgent:
    script: list[dict] | None = None

    def __init__(self, tools, model, response_model, reasoning_level):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level

    def chat(self, inputs=None, agent_state=None, return_usage=False, logger=None):
        if inputs is None:
            inputs = []
        if self.script is None:
            raise ValueError("FakeCodegenAgent.script must be set for scripted tool calls.")
        output = self._run_script(inputs)
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

    def _run_script(self, inputs):
        output = None
        for step in self.script or []:
            if "function_call" in step:
                call = step["function_call"] or {}
                name = call.get("name")
                params = call.get("params")
                if not name:
                    raise ValueError("Scripted function_call missing name.")
                tool_found = False
                for tool in self.tools:
                    if getattr(tool, "__name__", None) == name:
                        tool_found = True
                        if params is None:
                            tool()
                        else:
                            tool(params)
                        inputs.append({"type": "function_call_output", "output": {"ok": True}})
                        break
                if not tool_found:
                    raise ValueError(f"Tool '{name}' not found in FakeCodegenAgent.tools")
            if "output" in step:
                output = self._normalize_output(step["output"])
        return output

    def _normalize_output(self, output):
        if output is None:
            return None
        if self.response_model is None:
            return output
        if isinstance(output, self.response_model):
            return output
        if isinstance(output, dict):
            try:
                return self.response_model(**output)
            except Exception:
                return output
        return output
