from __future__ import annotations

import hashlib
import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.strategy import SearchStrategy
from pydantic import BaseModel, Field

from exps.mapping import build_doc_id_lookup, doc_ids_to_indices
from exps.run_dirs import dataset_from_trace_path, slugify
from exps.agentic.examples import append_few_shot_examples
from exps.agentic.task import build_task_tool
from exps.tools import (
    build_search_tools,
    normalize_search_tools,
    normalize_search_tools_for_cache,
)


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


def _parse_workflow(workflow: list) -> list[tuple[str, str]]:
    steps: list[tuple[str, str]] = []
    for entry in workflow:
        if isinstance(entry, dict) and len(entry) == 1:
            name, prompt = next(iter(entry.items()))
            if not isinstance(prompt, str):
                raise ValueError("Workflow prompts must be strings.")
            steps.append((name, prompt))
            continue
        raise ValueError("Workflow entries must be single-key mappings.")
    return steps


class SearchResults(BaseModel):
    """The state of the search agent, which can be used to inform future reasoning and tool use."""
    ranked_results: list[str] = Field(
        description="Top ranked search results (their doc_ids) when complete"
    )



def search(
    tools: list[callable] | None = None,
    inputs: list[dict] | None = None,
    agent_state: Optional[dict] = None,
    model: str = "gpt-5",
    text_format=SearchResults,
    reasoning: str = "medium",
):
    tools = tools or []
    inputs = inputs or []
    if agent_state is None:
        agent_state = {}
    agent = OpenAIAgent(
        tools=tools,
        model=f"openai/{model}" if "/" not in model else model,
        response_model=text_format,
        reasoning_level=reasoning,
    )
    return agent.loop(inputs=inputs, agent_state=agent_state)


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


def _parse_stop_entry(entry: Any) -> tuple[str, dict]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and len(entry) == 1:
        (name, raw_params), = entry.items()
        if isinstance(raw_params, dict):
            return name, dict(raw_params)
        if raw_params is None:
            return name, {}
        if name == "iterations":
            return name, {"iterations": raw_params}
        if name == "tool_calls":
            return name, {"tool_calls": raw_params}
        return name, {"value": raw_params}
    raise ValueError("Stop entry must be a string or single-key mapping.")


def normalize_stops(stop_config: list | None) -> list[dict[str, Any]]:
    if not stop_config:
        return []
    stops: list[dict[str, Any]] = []
    for entry in stop_config:
        name, params = _parse_stop_entry(entry)
        if name == "iterations" and "iterations" not in params:
            raise ValueError("Stop condition 'iterations' requires an iterations value.")
        if name == "tool_calls" and "tool_calls" not in params:
            raise ValueError("Stop condition 'tool_calls' requires a tool_calls value.")
        if name not in {"iterations", "tool_calls"}:
            raise ValueError(f"Unknown stop condition: {name}")
        stops.append({"name": name, "params": params})
    return stops


def _tool_calls_from_inputs(inputs: list) -> int:
    count = 0
    for item in inputs:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            count += 1
    return count


class AgenticSearchStrategy(SearchStrategy):
    _type = "agentic"

    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5-mini",
        reasoning: str = "medium",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list | None = None,
        subagent_system_prompt: str = SUBAGENT_SYSTEM_PROMPT,
        agents: dict | None = None,
        workflow: list | None = None,
        stop: list | None = None,
        reprompt: str | None = None,
        embeddings_device: str | None = None,
        trace_path: Path | None = None,
    ):
        self.embeddings_device = embeddings_device
        self.trace_path = trace_path
        self.dataset_name = dataset_from_trace_path(trace_path) if trace_path else None
        self.search_tools = search_tools or ["bm25"]
        self.agents = agents
        self.workflow = workflow
        if self.workflow and not self.agents:
            raise ValueError("workflow requires agents configuration.")
        self.subagent_system_prompt = subagent_system_prompt
        self.stop = stop
        self.reprompt = reprompt
        tool_config, delegate_task = _extract_delegate_task(self.search_tools)
        self._delegate_task = delegate_task
        self.tools = build_search_tools(
            corpus,
            tool_config,
            embeddings_device=embeddings_device,
            dataset_name=self.dataset_name,
        )
        self.model = model
        self.reasoning = reasoning
        self.system_prompt = system_prompt
        self._lookup = build_doc_id_lookup(corpus)
        self.traces: dict[str, str] = {}
        self.num_tool_calls: dict[str, int] = {}
        super().__init__(corpus, workers=workers)

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        device: str | None = None,
        **kwargs,
    ):
        build_params = dict(params)
        few_shot = build_params.pop("few_shot", None)
        if device and "embeddings_device" not in build_params:
            tool_config = build_params.get("search_tools") or ["bm25"]
            tool_names = [tool["name"] for tool in normalize_search_tools(tool_config)]
            if "minilm" in tool_names:
                build_params["embeddings_device"] = device
        if few_shot:
            system_prompt = build_params.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
            build_params["system_prompt"] = append_few_shot_examples(
                system_prompt,
                corpus=corpus,
                judgments=kwargs.get("judgments"),
                few_shot_config=few_shot,
            )
        return cls(corpus, workers=workers, **build_params)

    def _run_workflow_agent(
        self,
        *,
        agent_name: str,
        step_index: int,
        system_prompt: str,
        user_prompt: str,
        tool_config: list,
        inputs: list[dict],
        agent_state: dict,
        stops: list[dict[str, Any]],
        reprompt: str | None,
        logger,
    ):
        _replace_system_prompt(inputs, system_prompt)
        inputs.append({"role": "user", "content": user_prompt})

        filtered_tools, delegate_task = _extract_delegate_task(tool_config)
        if filtered_tools:
            step_tools_list = list(
                build_search_tools(
                    self.corpus,
                    filtered_tools,
                    embeddings_device=self.embeddings_device,
                    dataset_name=self.dataset_name,
                )
            )
        else:
            step_tools_list = []
        if delegate_task:
            task_tool = build_task_tool(
                search_tools=step_tools_list,
                model=self.model,
                reasoning=self.reasoning,
                system_prompt=self.subagent_system_prompt,
            )
            step_tools_list.insert(0, task_tool)

        agent = OpenAIAgent(
            tools=step_tools_list,
            model=f"openai/{self.model}" if "/" not in self.model else self.model,
            response_model=SearchResults,
            reasoning_level=self.reasoning,
        )

        num_loops = 0
        baseline_tool_calls = _tool_calls_from_inputs(inputs)
        while True:
            resp, inputs, _ = agent.chat(inputs=inputs, agent_state=agent_state, logger=logger)
            num_loops += 1
            tool_calls = _tool_calls_from_inputs(inputs) - baseline_tool_calls
            agent_state["num_tool_calls"] = _tool_calls_from_inputs(inputs)
            if not stops:
                break
            if any(
                (stopper["name"] == "iterations" and num_loops >= stopper["params"]["iterations"])
                or (
                    stopper["name"] == "tool_calls"
                    and tool_calls >= stopper["params"]["tool_calls"]
                )
                for stopper in stops
            ):
                break
            if reprompt:
                inputs.append({"role": "user", "content": reprompt})

        logger.info("agent_step_output %s", {"step": step_index, "agent": agent_name, "output": resp.output_parsed})
        return resp, inputs

    def _run_workflow(
        self,
        *,
        query: str,
        inputs: list[dict],
        agent_state: dict,
        stops: list[dict[str, Any]],
        reprompt: str | None,
        logger,
    ):
        if self.workflow:
            workflow_steps = _parse_workflow(self.workflow)
        else:
            workflow_steps = [("default", "The user's query: {query}")]

        resp = None
        for step_index, (agent_name, prompt_template) in enumerate(workflow_steps, start=1):
            logger.info("agent_step_start %s", {"step": step_index, "agent": agent_name})
            if self.workflow:
                agent_cfg = (self.agents or {}).get(agent_name)
                if agent_cfg is None:
                    raise ValueError(f"Unknown workflow agent: {agent_name}")
                step_system_prompt = agent_cfg.get("system_prompt", self.system_prompt)
                step_tool_config = agent_cfg.get("search_tools") or []
            else:
                step_system_prompt = self.system_prompt
                step_tool_config = self.search_tools

            user_prompt = prompt_template.format(query=query)
            resp, inputs = self._run_workflow_agent(
                agent_name=agent_name,
                step_index=step_index,
                system_prompt=step_system_prompt,
                user_prompt=user_prompt,
                tool_config=step_tool_config,
                inputs=inputs,
                agent_state=agent_state,
                stops=stops,
                reprompt=reprompt,
                logger=logger,
            )
        return resp, inputs

    def search(self, query: str, k: int = 10):
        if self.trace_path is None:
            raise ValueError("AgenticSearchStrategy requires trace_path to record traces.")
        query_dir = self.query_path(query)
        inputs = [{"role": "system", "content": self.system_prompt}]
        agent_state = {"num_tool_calls": 0}
        stops = normalize_stops(self.stop)
        reprompt = self.reprompt
        if reprompt is not None and not isinstance(reprompt, str):
            raise ValueError("reprompt must be a string when provided.")
        with trace_logger(query_dir) as (logger, trace_path):
            logger.info("Query: %s", query)
            agent_state["trace_logger"] = logger
            agent_state["run_dir"] = str(query_dir)
            resp, _ = self._run_workflow(
                query=query,
                inputs=inputs,
                agent_state=agent_state,
                stops=stops,
                reprompt=reprompt,
                logger=logger,
            )

            ranked_results = (resp.output_parsed.ranked_results or [])[:k] if resp else []
            logger.info("agentic_output %s", resp.output_parsed if resp else None)
            if self._lookup:
                ranked_results = doc_ids_to_indices(ranked_results, self._lookup)
        self.traces[query] = str(trace_path)
        num_tool_calls = int(agent_state.get("num_tool_calls", 0))
        self.num_tool_calls[query] = num_tool_calls
        summary_path = query_dir / "summary.json"
        summary_path.write_text(
            json.dumps({"num_tool_calls": num_tool_calls}, indent=2) + "\n",
            encoding="utf-8",
        )
        return ranked_results, [1.0] * len(ranked_results)

    def query_path(self, query: str) -> Path:
        if self.trace_path is None:
            raise ValueError("AgenticSearchStrategy requires trace_path to record traces.")
        query_slug = slugify(query, fallback="query")
        query_dir = self.trace_path / query_slug
        if query_dir.exists():
            counter = 2
            while True:
                candidate = self.trace_path / f"{query_slug}_{counter}"
                try:
                    candidate.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    counter += 1
                    continue
                query_dir = candidate
                break
        return query_dir

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
            "search_tools": _normalize_search_tools_for_cache(self.search_tools),
            "subagent_system_prompt": self.subagent_system_prompt,
            "agents": _normalize_agents_for_cache(self.agents),
            "workflow": self.workflow,
            "stop": self.stop,
            "reprompt": self.reprompt,
            "embeddings_device": self.embeddings_device,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
