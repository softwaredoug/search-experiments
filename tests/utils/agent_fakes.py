from __future__ import annotations

import copy
from types import SimpleNamespace

from exps.agentic import conditions as conditions_mod
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
    def __init__(self, tools, model, response_model, reasoning_level, images=False, process_images=False):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level
        self.process_images = process_images or images
        self.calls = 0
        self.chat_calls = 0
        self._script_index = 0
        self.doc_ids: list[str] = []
        self.categories: list[str] = []
        self.scripts: list[list[dict]] | None = None

    def loop(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        self.calls += 1
        self.chat_calls += 1
        if agent_state is not None:
            agent_state["resource.input_tokens"] = (
                agent_state.get("resource.input_tokens", 0) + 1
            )
            agent_state["resource.output_tokens"] = (
                agent_state.get("resource.output_tokens", 0) + 2
            )
        script = self._next_script()
        output = self._run_script(inputs, script, agent_state)
        resp = type("Resp", (), {"output_parsed": output})
        self.last_inputs = copy.deepcopy(inputs)
        return resp, inputs

    def _build_output(self):
        model = self.response_model
        if model is None:
            return None
        fields = getattr(model, "model_fields", None)
        if fields and "categories" in fields:
            return model(categories=list(self.categories))
        if fields and "doc_ids_ranked" in fields:
            return model(doc_ids_ranked=list(self.doc_ids))
        try:
            return model()
        except TypeError:
            return SearchResults(doc_ids_ranked=list(self.doc_ids))

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

    def _run_script(self, inputs, script, agent_state):
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
                        tool_output = self._call_tool(tool, params, agent_state)
                        if tool_output is None:
                            tool_output = {"ok": True}
                        inputs.append({"type": "function_call_output", "output": tool_output})
                        break
                if not tool_found:
                    raise ValueError(f"Tool '{name}' not found in FakeOpenAIAgent.tools")
            if "output" in step:
                output = self._normalize_output(step["output"])
        return output

    def _call_tool(self, tool, params, agent_state):
        if params is None:
            try:
                return tool(agent_state=agent_state)
            except TypeError:
                return tool()
        if isinstance(params, dict):
            try:
                return tool(**params, agent_state=agent_state)
            except TypeError:
                return tool(**params)
        try:
            return tool(params, agent_state=agent_state)
        except TypeError:
            return tool(params)

    def _normalize_output(self, output):
        if output is None:
            return self._build_output()
        if self.response_model is None:
            return output
        if isinstance(output, self.response_model):
            return output
        if isinstance(output, dict):
            if "ranked_results" in output and "doc_ids_ranked" not in output:
                output = dict(output)
                output["doc_ids_ranked"] = output.pop("ranked_results")
            try:
                return self.response_model(**output)
            except Exception:
                return output
        return output


class FakeLLMJudgeAgent:
    calls = 0
    emojis = ["😞", "😃"]

    def __init__(self, tools, model, response_model, reasoning_level, process_images=False):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level
        self.process_images = process_images

    @classmethod
    def reset(cls, emojis: list[str] | None = None) -> None:
        cls.calls = 0
        if emojis is not None:
            cls.emojis = list(emojis)

    def loop(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        emoji_index = min(self.calls, max(len(self.emojis) - 1, 0))
        emoji = self.emojis[emoji_index]
        type(self).calls += 1
        result = conditions_mod.LLMJudgeResponse(
            graded_results=[
                conditions_mod.GradedSearchResult(
                    emoji=emoji,
                    title="Sample",
                    doc_id="101",
                )
            ]
        )
        resp = type("Resp", (), {"output_parsed": result})
        return resp, inputs


class FakeJevChoice:
    def __init__(self, *, instructions, criteria):
        self.instructions = instructions
        self.criteria = criteria


class FakeJevClient:
    responses: list[tuple[str, float | None, float | None]] = []
    calls: list[tuple[str, FakeJevChoice]] = []
    clients: list["FakeJevClient"] = []
    response_index = 0

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        type(self).clients.append(self)

    @classmethod
    def reset(cls, responses=None):
        cls.responses = list(responses or [])
        cls.calls = []
        cls.clients = []
        cls.response_index = 0

    def system_one(self, *, state, questions):
        field, choice = next(iter(questions.items()))
        type(self).calls.append((state, choice))
        if not type(self).responses:
            raise AssertionError("FakeJevClient.reset() requires at least one response.")
        label, probability, confidence = type(self).responses[
            min(type(self).response_index, len(type(self).responses) - 1)
        ]
        type(self).response_index += 1
        answer = SimpleNamespace(
            choice=label,
            probabilities={label: probability} if probability is not None else None,
            confidence=confidence,
        )
        return SimpleNamespace(choices={field: answer})


class FakeAutoDecisionEnricher:
    instances = []

    def __init__(self, *, choices, model, **_kwargs):
        self.choices = dict(choices)
        self.model = model
        provider, _, model_name = model.partition("/")
        self.client = FakeJevClient(
            api_key="test-key",
            model=model_name if provider == "jev" else model,
        )
        type(self).instances.append(self)

    def score(self, *, state, question):
        answer = self.client.system_one(
            state=state,
            questions={
                "relevance": FakeJevChoice(
                    instructions=question,
                    criteria=self.choices,
                )
            },
        ).choices["relevance"]
        return answer.probabilities or {}, answer.confidence


class FakeCodegenAgent:
    script: list[dict] | None = None

    def __init__(self, tools, model, response_model, reasoning_level, images=False):
        self.tools = tools
        self.model = model
        self.response_model = response_model
        self.reasoning_level = reasoning_level

    def loop(self, inputs=None, agent_state=None, logger=None):
        if inputs is None:
            inputs = []
        if self.script is None:
            raise ValueError("FakeCodegenAgent.script must be set for scripted tool calls.")
        output = self._run_script(inputs)
        resp = type("Resp", (), {"output_parsed": output})
        return resp, inputs

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
