from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Type

from cheat_at_search.agent.openai_agent import OpenAIAgent
from pydantic import BaseModel, Field

from exps.agentic.conditions import evaluate_stopper, evaluate_validator, normalize_conditions
from exps.agentic.task import build_task_tool
from exps.mapping import build_doc_id_lookup
from exps.tools import build_search_tools, normalize_search_tools_for_cache


DEFAULT_SYSTEM_PROMPT = """
You take user search queries and use a search tool to find furniture / home goods products.

Look at the search tools you have, their limitations, how they work, etc when forming your plan.

Finally return results to the user per the SearchResults schema, ranked best to worst.

Gather results until you have 10 best matches you can find. It's important to return at least 10.

Consider possibly

* Not searching categories if no relevant results found

It's very important you consider carefully the correct ranking as you'll be evaluated on
how close that is to the average furniture shoppers ideal ranking.

Here are some examples of products and relevant / irrelevant results
"""


SUBAGENT_SYSTEM_PROMPT = "You help with tasks searchinging / finding content as instructed"


class SearchResults(BaseModel):
    """The state of the search agent, which can be used to inform future reasoning and tool use."""

    ranked_results: list[str] = Field(
        description="Top ranked search results (their doc_ids) when complete"
    )


@dataclass
@dataclass
class AgentResponse:
    output: BaseModel | list[str] | None
    trace_path: Path
    num_tool_calls: int


def _extract_delegate_task(tool_config: list | None) -> tuple[list, bool]:
    if not tool_config:
        return [], False
    remaining: list = []
    delegate_task = False
    for entry in tool_config:
        if isinstance(entry, str) and entry == "delegate_task":
            delegate_task = True
            continue
        if isinstance(entry, dict) and len(entry) == 1 and "delegate_task" in entry:
            delegate_task = True
            continue
        remaining.append(entry)
    return remaining, delegate_task


def _normalize_search_tools_for_cache(tool_config: list) -> list[dict[str, Any]]:
    filtered, delegate_task = _extract_delegate_task(tool_config)
    normalized = normalize_search_tools_for_cache(filtered)
    if delegate_task:
        normalized.append({"name": "delegate_task", "guards": [], "config": {}})
    return normalized


def _normalize_agents_for_cache(agents: dict[str, dict] | None) -> dict[str, Any] | None:
    if not agents:
        return None
    payload: dict[str, Any] = {}
    for name, config in agents.items():
        payload[name] = {
            "system_prompt": config.get("system_prompt"),
            "search_tools": _normalize_search_tools_for_cache(config.get("search_tools") or []),
        }
    return payload


def _replace_system_prompt(inputs: list[dict], system_prompt: str) -> None:
    for item in inputs:
        if isinstance(item, dict) and item.get("role") == "system":
            item["content"] = system_prompt
            return
    inputs.insert(0, {"role": "system", "content": system_prompt})


def _parse_plan(plan: list) -> list[tuple[str, str]]:
    steps: list[tuple[str, str]] = []
    for entry in plan:
        if isinstance(entry, dict) and len(entry) == 1:
            name, prompt = next(iter(entry.items()))
            if not isinstance(prompt, str):
                raise ValueError("Plan prompts must be strings.")
            steps.append((name, prompt))
            continue
        raise ValueError("Plan entries must be single-key mappings.")
    return steps


@contextmanager
def trace_logger(trace_dir: Path):
    trace_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trace_path = trace_dir / f"{timestamp}.log"
    logger = logging.getLogger(f"agentic.trace.{trace_dir.name}.{timestamp}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(trace_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield logger, trace_path
    finally:
        handler.close()
        logger.removeHandler(handler)


def _tool_calls_from_inputs(inputs: list) -> int:
    count = 0
    for item in inputs:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            count += 1
    return count


class Agent:
    def __init__(
        self,
        *,
        corpus,
        model: str = "gpt-5-mini",
        reasoning: str = "medium",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list | None = None,
        subagent_system_prompt: str = SUBAGENT_SYSTEM_PROMPT,
        agents: dict | None = None,
        plan: list | None = None,
        stop: list | None = None,
        validators: list | None = None,
        max_loops: int = 10,
        embeddings_device: str | None = None,
        dataset_name: str | None = None,
        response_model: Type[BaseModel] | None = SearchResults,
    ):
        self.corpus = corpus
        self.model = model
        self.reasoning = reasoning
        self.system_prompt = system_prompt
        self.search_tools = search_tools or ["bm25"]
        self.subagent_system_prompt = subagent_system_prompt
        self.agents = agents
        self.plan = plan
        if self.plan and not self.agents:
            raise ValueError("plan requires agents configuration.")
        self.stop = stop
        self.validators = validators
        self.max_loops = max_loops
        self.embeddings_device = embeddings_device
        self.dataset_name = dataset_name
        self.response_model = response_model
        self._lookup = build_doc_id_lookup(corpus)
        self._tool_cache: dict[str, list[callable]] = {}
        self._plan_steps = self._prepare_plan_steps()

    @property
    def lookup(self):
        return self._lookup

    def _prepare_plan_steps(self) -> list[dict[str, Any]]:
        if self.plan:
            plan_steps = _parse_plan(self.plan)
        else:
            plan_steps = [("default", "The user's query: {query}")]
        prepared = []
        for agent_name, prompt_template in plan_steps:
            if self.plan:
                agent_cfg = (self.agents or {}).get(agent_name)
                if agent_cfg is None:
                    raise ValueError(f"Unknown plan agent: {agent_name}")
                step_system_prompt = agent_cfg.get("system_prompt", self.system_prompt)
                step_tool_config = agent_cfg.get("search_tools") or []
            else:
                step_system_prompt = self.system_prompt
                step_tool_config = self.search_tools
            prepared.append(
                {
                    "agent_name": agent_name,
                    "prompt_template": prompt_template,
                    "system_prompt": step_system_prompt,
                    "tools": self._get_tools(step_tool_config, system_prompt=step_system_prompt),
                }
            )
        return prepared

    def _get_tools(self, tool_config: list | None, *, system_prompt: str | None = None) -> list[callable]:
        if not tool_config:
            return []
        cache_key = json.dumps(
            {
                "tools": _normalize_search_tools_for_cache(tool_config),
                "system_prompt": system_prompt,
            },
            sort_keys=True,
            default=str,
        )
        cached = self._tool_cache.get(cache_key)
        if cached is not None:
            return cached
        filtered_tools, delegate_task = _extract_delegate_task(tool_config)
        if filtered_tools:
            tools = list(
                build_search_tools(
                    self.corpus,
                    filtered_tools,
                    embeddings_device=self.embeddings_device,
                    dataset_name=self.dataset_name,
                    system_prompt=system_prompt,
                )
            )
        else:
            tools = []
        if delegate_task:
            task_tool = build_task_tool(
                search_tools=tools,
                model=self.model,
                reasoning=self.reasoning,
                system_prompt=self.subagent_system_prompt,
            )
            tools.insert(0, task_tool)
        self._tool_cache[cache_key] = tools
        return tools

    def _run_plan_agent(
        self,
        *,
        agent_name: str,
        step_index: int,
        query: str,
        system_prompt: str,
        user_prompt: str,
        tools: list[callable],
        inputs: list[dict],
        agent_state: dict,
        stops: list[dict[str, Any]],
        validators: list[dict[str, Any]],
        logger,
    ):
        _replace_system_prompt(inputs, system_prompt)
        inputs.append({"role": "user", "content": user_prompt})
        step_tools_list = list(tools)

        agent = OpenAIAgent(
            tools=step_tools_list,
            model=f"openai/{self.model}" if "/" not in self.model else self.model,
            response_model=self.response_model,
            reasoning_level=self.reasoning,
        )

        num_loops = 0
        baseline_tool_calls = _tool_calls_from_inputs(inputs)
        while True:
            resp, inputs, _ = agent.chat(inputs=inputs, agent_state=agent_state, logger=logger)
            num_loops += 1
            tool_calls = _tool_calls_from_inputs(inputs) - baseline_tool_calls
            agent_state["num_tool_calls"] = _tool_calls_from_inputs(inputs)
            if num_loops >= self.max_loops:
                break
            for validator in validators:
                result = evaluate_validator(
                    validator,
                    num_loops=num_loops,
                    tool_calls=tool_calls,
                    resp=resp,
                    query=query,
                    corpus=self.corpus,
                    lookup=self._lookup,
                    logger=logger,
                )
                if result is True:
                    continue
                logger.info(
                    "agentic_validator_prompt %s",
                    {
                        "step": step_index,
                        "agent": agent_name,
                        "validator": validator["name"],
                        "result": result,
                    },
                )
                inputs.append({"role": "user", "content": result})
                break
            else:
                if not stops:
                    break
                stop_prompt = None
                for stopper in stops:
                    stop_result = evaluate_stopper(
                        stopper,
                        num_loops=num_loops,
                        tool_calls=tool_calls,
                        resp=resp,
                    )
                    if stop_result is True:
                        stop_prompt = None
                        break
                    if stop_prompt is None:
                        stop_prompt = stop_result
                        logger.info(
                            "agentic_stopper_prompt %s",
                            {
                                "step": step_index,
                                "agent": agent_name,
                                "stopper": stopper["name"],
                                "result": stop_result,
                            },
                        )
                if stop_prompt is None:
                    break
                inputs.append({"role": "user", "content": stop_prompt})

        logger.info("agent_step_output %s", {"step": step_index, "agent": agent_name, "output": resp.output_parsed})
        return resp, inputs

    def _run_plan(
        self,
        *,
        query: str,
        inputs: list[dict],
        agent_state: dict,
        stops: list[dict[str, Any]],
        validators: list[dict[str, Any]],
        logger,
        format_params: dict[str, Any],
    ):
        resp = None
        for step_index, step in enumerate(self._plan_steps, start=1):
            logger.info(
                "agent_step_start %s",
                {"step": step_index, "agent": step["agent_name"]},
            )
            user_prompt = step["prompt_template"].format(**format_params)
            resp, inputs = self._run_plan_agent(
                agent_name=step["agent_name"],
                step_index=step_index,
                query=query,
                system_prompt=step["system_prompt"],
                user_prompt=user_prompt,
                tools=step["tools"],
                inputs=inputs,
                agent_state=agent_state,
                stops=stops,
                validators=validators,
                logger=logger,
            )
        return resp, inputs

    def run(
        self,
        *,
        query: str,
        trace_dir: Path,
        k: int = 10,
        format_params: dict[str, Any] | None = None,
        logger=None,
        trace_path: Path | None = None,
    ) -> AgentResponse:
        inputs = [{"role": "system", "content": self.system_prompt}]
        agent_state = {"num_tool_calls": 0}
        stops = normalize_conditions(self.stop, kind="stop")
        validators = normalize_conditions(self.validators, kind="validator")
        format_payload = {"query": query}
        if format_params:
            format_payload.update(format_params)
        if logger is None:
            with trace_logger(trace_dir) as (logger, trace_path):
                logger.info("Query: %s", query)
                agent_state["trace_logger"] = logger
                agent_state["run_dir"] = str(trace_dir)
                resp, _ = self._run_plan(
                    query=query,
                    inputs=inputs,
                    agent_state=agent_state,
                    stops=stops,
                    validators=validators,
                    logger=logger,
                    format_params=format_payload,
                )
                logger.info("agentic_output %s", resp.output_parsed if resp else None)
                if resp and hasattr(resp.output_parsed, "ranked_results"):
                    ranked_results = resp.output_parsed.ranked_results or []
                    logger.info(
                        "agentic_complete %s",
                        {"query": query, "results": len(ranked_results)},
                    )
        else:
            if trace_path is None:
                raise ValueError("trace_path is required when passing a logger.")
            logger.info("Query: %s", query)
            agent_state["trace_logger"] = logger
            agent_state["run_dir"] = str(trace_dir)
            resp, _ = self._run_plan(
                query=query,
                inputs=inputs,
                agent_state=agent_state,
                stops=stops,
                validators=validators,
                logger=logger,
                format_params=format_payload,
            )
            logger.info("agentic_output %s", resp.output_parsed if resp else None)
            if resp and hasattr(resp.output_parsed, "ranked_results"):
                ranked_results = resp.output_parsed.ranked_results or []
                logger.info(
                    "agentic_complete %s",
                    {"query": query, "results": len(ranked_results)},
                )
        num_tool_calls = int(agent_state.get("num_tool_calls", 0))
        output = resp.output_parsed if resp else None
        if output and hasattr(output, "ranked_results"):
            output = list(output.ranked_results or [])[:k]
        return AgentResponse(
            output=output,
            trace_path=trace_path,
            num_tool_calls=num_tool_calls,
        )
