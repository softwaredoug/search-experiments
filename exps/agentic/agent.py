from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from time import monotonic
from typing import Any, Type

from cheat_at_search.agent.openai_agent import OpenAIAgent
from cheat_at_search.agent.search_client import SearchResults
from cheat_at_search.agent.sid_agent import SIDAgent
from cheat_at_search.agent.sid_legacy_agent import SIDLegacyAgent
from pydantic import BaseModel, Field, create_model

from exps.agentic.conditions import (
    Condition,
    evaluate_stopper,
    evaluate_validator,
    normalize_conditions,
)
from exps.agentic.task import build_task_tool
from exps.agentic.tracing import query_heartbeat, set_trace_stage, trace_event
from exps.mapping import build_doc_id_lookup
from exps.tools import build_search_tools, normalize_search_tools_for_cache
from exps.resource_tracking import resources_from_agent_state


DEFAULT_SYSTEM_PROMPT = """
You take user search queries and use a search tool to find furniture / home goods products.

Look at the search tools you have, their limitations, how they work, etc when forming your plan.

Finally return exactly 10 results to the user per the SearchResults schema, ranked best to worst.

Gather and return exactly 10 best matches when available.

Consider possibly

* Not searching categories if no relevant results found

It's very important you consider carefully the correct ranking as you'll be evaluated on
how close that is to the average furniture shoppers ideal ranking.

Here are some examples of products and relevant / irrelevant results
"""


SUBAGENT_SYSTEM_PROMPT = "You help with tasks searchinging / finding content as instructed"


@lru_cache(maxsize=None)
def search_results_model_for_k(k: int) -> Type[BaseModel]:
    """Build a SearchResults schema requiring exactly k ranked document IDs."""
    if k < 1:
        return SearchResults
    return create_model(
        f"SearchResultsForK{k}",
        __base__=SearchResults,
        doc_ids_ranked=(
            list[str],
            Field(
                min_length=k,
                max_length=k,
                description=SearchResults.model_fields["doc_ids_ranked"].description,
            ),
        ),
    )


@dataclass
@dataclass
class AgentResponse:
    output: BaseModel | list[str] | None
    trace_path: Path
    num_tool_calls: int
    resources: dict[str, int | float]


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
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call_output" or (
            item.get("role") == "tool" and "tool_call_id" in item
        ):
            count += 1
    return count


def build_openai_agent(
    *,
    tools: list[callable],
    model: str,
    response_model: Type[BaseModel] | None,
    reasoning_level: str,
    images: bool = False,
) -> OpenAIAgent:
    return OpenAIAgent(
        tools=tools,
        model=model,
        response_model=response_model,
        reasoning_level=reasoning_level,
        process_images=images,
    )


def _adapt_sid2_result(result):
    """Give SID-2 the string IDs and content field its document cache requires."""
    if not isinstance(result, list):
        return result
    adapted = []
    for item in result:
        if not isinstance(item, dict):
            adapted.append(item)
            continue
        document = dict(item)
        if document.get("id") is not None:
            document["id"] = str(document["id"])
        content = document.get("content")
        if not isinstance(content, str) or not content:
            content = document.get("description")
        if not isinstance(content, str) or not content:
            content = document.get("title")
        if isinstance(content, str) and content:
            document["content"] = content
        adapted.append(document)
    return adapted


def _adapt_tool_for_sid2(tool: callable) -> callable:
    @wraps(tool)
    def sid2_tool(*args, **kwargs):
        return _adapt_sid2_result(tool(*args, **kwargs))

    aliases = getattr(tool, "aliases", None)
    if aliases:
        # This cheat-at-search revision checks __name__ when deciding whether
        # SID-2 has searched, even though schemas and dispatch use aliases.
        sid2_tool.__name__ = aliases[0]
    return sid2_tool


def _adapt_tools_for_sid2(tools: list[callable]) -> list[callable]:
    """Adapt standard search documents to SID-2's cached document format."""
    adapted_tools = []
    used_names = set()
    for tool in tools:
        adapted_tool = _adapt_tool_for_sid2(tool)
        aliases = getattr(tool, "aliases", None)
        names = list(aliases) if aliases else [tool.__name__]
        if aliases and used_names.intersection(names):
            # SID-2 exposes each alias as a separate function name, so retain
            # the native name for later tools in the same search family.
            adapted_tool.aliases = []
            adapted_tool.__name__ = tool.__name__
            names = [tool.__name__]
        elif aliases:
            # The pinned SIDAgent checks __name__ to recognize a search call.
            adapted_tool.__name__ = aliases[0]
        used_names.update(names)
        adapted_tools.append(adapted_tool)
    return adapted_tools


class Agent:
    def __init__(
        self,
        *,
        corpus,
        judgments=None,
        model: str = "gpt-5-mini",
        reasoning: str = "medium",
        images: bool = False,
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
        self.images = images
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
        self.judgments = judgments
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
                "images": self.images,
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
                images=self.images,
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
        stops: list[Condition],
        validators: list[Condition],
        logger,
        response_model: Type[BaseModel] | None = None,
    ):
        _replace_system_prompt(inputs, system_prompt)
        inputs.append({"role": "user", "content": user_prompt})
        step_tools_list = list(tools)

        set_trace_stage(
            agent_state,
            "agent_setup",
            step=step_index,
            agent=agent_name,
        )
        setup_started = monotonic()
        trace_event(
            logger,
            "agentic_agent_setup_start",
            step=step_index,
            agent=agent_name,
            tool_count=len(step_tools_list),
        )
        provider, separator, provider_model = self.model.partition("/")
        if separator and provider == "sid":
            sid_agents = {
                "sid-1": SIDLegacyAgent,
                "sid-2": SIDAgent,
            }
            sid_agent_class = sid_agents.get(provider_model)
            if sid_agent_class is None:
                raise ValueError(
                    f"Unsupported SID model {provider_model!r}; expected 'sid-1' or 'sid-2'."
                )
            sid_agent_params = {
                "tools": step_tools_list,
                "max_turns": self.max_loops,
                "model": provider_model,
            }
            if provider_model == "sid-2":
                if self.reasoning not in {"low", "medium", "high"}:
                    raise ValueError(
                        "SID-2 reasoning must be one of 'low', 'medium', or 'high'."
                    )
                step_tools_list = _adapt_tools_for_sid2(step_tools_list)
                sid_agent_params["tools"] = step_tools_list
                sid_agent_params["reasoning_effort"] = self.reasoning
            agent = sid_agent_class(**sid_agent_params)
        else:
            agent = build_openai_agent(
                tools=step_tools_list,
                model=f"openai/{self.model}" if not separator else self.model,
                response_model=(
                    self.response_model if response_model is None else response_model
                ),
                reasoning_level=self.reasoning,
                images=self.images,
            )
        trace_event(
            logger,
            "agentic_agent_setup_complete",
            step=step_index,
            agent=agent_name,
            elapsed_seconds=round(monotonic() - setup_started, 3),
        )

        num_loops = 0
        baseline_tool_calls = _tool_calls_from_inputs(inputs)
        while True:
            chat_started = monotonic()
            set_trace_stage(
                agent_state,
                "agent_chat",
                step=step_index,
                agent=agent_name,
                turn=num_loops + 1,
            )
            trace_event(
                logger,
                "agentic_chat_start",
                step=step_index,
                agent=agent_name,
                turn=num_loops + 1,
            )
            try:
                resp, inputs = agent.loop(
                    inputs=inputs,
                    agent_state=agent_state,
                    logger=logger,
                )
            except Exception as exc:
                trace_event(
                    logger,
                    "agentic_chat_error",
                    step=step_index,
                    agent=agent_name,
                    turn=num_loops + 1,
                    elapsed_seconds=round(monotonic() - chat_started, 3),
                    error_type=type(exc).__name__,
                )
                raise
            num_loops += 1
            tool_calls = _tool_calls_from_inputs(inputs) - baseline_tool_calls
            agent_state["num_tool_calls"] = _tool_calls_from_inputs(inputs)
            trace_event(
                logger,
                "agentic_chat_complete",
                step=step_index,
                agent=agent_name,
                turn=num_loops,
                elapsed_seconds=round(monotonic() - chat_started, 3),
                total_tool_calls=agent_state["num_tool_calls"],
            )
            if num_loops >= self.max_loops:
                trace_event(
                    logger,
                    "agentic_max_loops_reached",
                    step=step_index,
                    agent=agent_name,
                    max_loops=self.max_loops,
                )
                break
            for validator in validators:
                validation_started = monotonic()
                set_trace_stage(
                    agent_state,
                    "validator",
                    validator=validator["name"],
                    turn=num_loops,
                )
                trace_event(
                    logger,
                    "agentic_validator_start",
                    validator=validator["name"],
                    turn=num_loops,
                )
                try:
                    result = evaluate_validator(
                        validator,
                        num_loops=num_loops,
                        tool_calls=tool_calls,
                        resp=resp,
                        query=query,
                        corpus=self.corpus,
                        lookup=self._lookup,
                        judgments=self.judgments,
                        agent_state=agent_state,
                        logger=logger,
                        images=self.images,
                    )
                except Exception as exc:
                    trace_event(
                        logger,
                        "agentic_validator_error",
                        validator=validator["name"],
                        turn=num_loops,
                        elapsed_seconds=round(monotonic() - validation_started, 3),
                        error_type=type(exc).__name__,
                    )
                    raise
                trace_event(
                    logger,
                    "agentic_validator_complete",
                    validator=validator["name"],
                    turn=num_loops,
                    elapsed_seconds=round(monotonic() - validation_started, 3),
                    outcome="passed" if result is True else "feedback",
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
                    set_trace_stage(
                        agent_state,
                        "stopper",
                        stopper=stopper["name"],
                        turn=num_loops,
                    )
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
        stops: list[Condition],
        validators: list[Condition],
        logger,
        format_params: dict[str, Any],
        response_model: Type[BaseModel] | None,
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
                response_model=response_model,
            )
        return resp, inputs

    def run(
        self,
        *,
        query: str,
        trace_dir: Path,
        logger,
        trace_path: Path,
        k: int = 10,
        format_params: dict[str, Any] | None = None,
        agent_state: dict | None = None,
        response_model: Type[BaseModel] | None = None,
    ) -> AgentResponse:
        inputs = [{"role": "system", "content": self.system_prompt}]
        if agent_state is None:
            agent_state = {}
        agent_state["num_tool_calls"] = 0
        stops = normalize_conditions(self.stop, kind="stop")
        validators = normalize_conditions(self.validators, kind="validator")
        format_payload = {"query": query}
        if format_params:
            format_payload.update(format_params)
        logger.info("Query: %s", query)
        agent_state["trace_logger"] = logger
        agent_state["run_dir"] = str(trace_dir)
        set_trace_stage(agent_state, "agent_run_start")
        trace_event(logger, "agentic_query_start", query=query)
        query_started = monotonic()
        with query_heartbeat(
            query=query,
            agent_state=agent_state,
            logger=logger,
        ):
            resp, _ = self._run_plan(
                query=query,
                inputs=inputs,
                agent_state=agent_state,
                stops=stops,
                validators=validators,
                logger=logger,
                format_params=format_payload,
                response_model=(
                    self.response_model if response_model is None else response_model
                ),
            )
        logger.info("agentic_output %s", resp.output_parsed if resp else None)
        if resp and hasattr(resp.output_parsed, "doc_ids_ranked"):
            ranked_results = resp.output_parsed.doc_ids_ranked or []
            logger.info(
                "agentic_complete %s",
                {"query": query, "results": len(ranked_results)},
            )
        trace_event(
            logger,
            "agentic_query_complete",
            query=query,
            elapsed_seconds=round(monotonic() - query_started, 3),
            tool_calls=agent_state.get("num_tool_calls", 0),
        )
        num_tool_calls = int(agent_state.get("num_tool_calls", 0))
        output = resp.output_parsed if resp else None
        if output and hasattr(output, "doc_ids_ranked"):
            output = list(output.doc_ids_ranked or [])[:k]
        return AgentResponse(
            output=output,
            trace_path=trace_path,
            num_tool_calls=num_tool_calls,
            resources=resources_from_agent_state(
                agent_state, tool_calls=num_tool_calls
            ),
        )
