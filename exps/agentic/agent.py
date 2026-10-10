from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Type

from cheat_at_search.agent.openai_agent import OpenAIAgent
from pydantic import BaseModel, Field

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


AGENTIC_RANKED_RESULTS_LENGTH = 10


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


class SearchResults(BaseModel):
    """The state of the search agent, which can be used to inform future reasoning and tool use."""

    ranked_results: list[str] = Field(
        description="Top ranked search results (their doc_ids) when complete"
    )


class AgenticSearchResults(SearchResults):
    """Standard agentic response schema with the fixed ten-result contract."""

    ranked_results: list[str] = Field(
        min_length=AGENTIC_RANKED_RESULTS_LENGTH,
        max_length=AGENTIC_RANKED_RESULTS_LENGTH,
        description="Exactly ten document IDs, ranked best to worst.",
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


def build_openai_agent(
    *,
    tools: list[callable],
    model: str,
    response_model: Type[BaseModel] | None,
    reasoning_level: str,
    images: bool = False,
) -> OpenAIAgent:
    return TracingOpenAIAgent(
        tools=tools,
        model=model,
        response_model=response_model,
        reasoning_level=reasoning_level,
        process_images=images,
    )


def _instrument_search_tool(tool_name: str, call_from_tool):
    def _timed_tool(args_model, agent_state=None):
        logger = agent_state.get("trace_logger") if agent_state else None
        call_index = 0
        if agent_state is not None:
            call_index = agent_state.get("_trace_tool_calls", 0) + 1
            agent_state["_trace_tool_calls"] = call_index
        started = monotonic()
        set_trace_stage(
            agent_state,
            "search_tool",
            tool=tool_name,
            call_index=call_index,
        )
        trace_event(
            logger,
            "agentic_tool_start",
            tool=tool_name,
            call_index=call_index,
        )
        try:
            result = call_from_tool(args_model, agent_state=agent_state)
        except Exception as exc:
            trace_event(
                logger,
                "agentic_tool_error",
                tool=tool_name,
                call_index=call_index,
                elapsed_seconds=round(monotonic() - started, 3),
                error_type=type(exc).__name__,
            )
            raise
        tool_error = (
            isinstance(result, tuple)
            and bool(result)
            and isinstance(result[0], str)
            and result[0].startswith("Tool error:")
        )
        trace_event(
            logger,
            "agentic_tool_complete",
            tool=tool_name,
            call_index=call_index,
            elapsed_seconds=round(monotonic() - started, 3),
            outcome="error" if tool_error else "ok",
        )
        set_trace_stage(
            agent_state,
            "agent_chat",
            last_tool=tool_name,
            call_index=call_index,
        )
        return result

    return _timed_tool


class _OpenAIRetryTraceLogger:
    def __init__(self, logger, request_index: int):
        self.logger = logger
        self.request_index = request_index
        self.retry_count = 0

    def warning(self, message, *args, **kwargs):
        self.retry_count += 1
        error = args[0] if args else None
        trace_event(
            self.logger,
            "agentic_openai_request_retry",
            request_index=self.request_index,
            attempt=self.retry_count + 1,
            error_type=type(error).__name__ if error is not None else "unknown",
        )
        self.logger.warning(message, *args, **kwargs)


class TracingOpenAIAgent(OpenAIAgent):
    """OpenAI agent that records each Responses API attempt in the query trace."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trace_request_index = 0
        for tool_name, (args_model, tool_spec, call_from_tool) in tuple(
            self.search_tools.items()
        ):
            self.search_tools[tool_name] = (
                args_model,
                tool_spec,
                _instrument_search_tool(tool_name, call_from_tool),
            )

    def chat(self, inputs=None, agent_state=None, return_usage=False, logger=None):
        self._active_agent_state = agent_state
        try:
            return super().chat(
                inputs=inputs,
                agent_state=agent_state,
                return_usage=return_usage,
                logger=logger,
            )
        finally:
            self._active_agent_state = None

    def _call_responses_with_retry(self, inputs, tools, reasoning, active_logger):
        # Delegate retry policy to cheat-at-search; its retry warnings are
        # mirrored as structured events with the logical request ID.
        self._trace_request_index += 1
        request_index = self._trace_request_index
        started = monotonic()
        set_trace_stage(
            getattr(self, "_active_agent_state", None),
            "openai_request",
            request_index=request_index,
        )
        trace_event(
            active_logger,
            "agentic_openai_request_start",
            request_index=request_index,
            model=self.model,
        )
        retry_logger = _OpenAIRetryTraceLogger(active_logger, request_index)
        try:
            response = super()._call_responses_with_retry(
                inputs=inputs,
                tools=tools,
                reasoning=reasoning,
                active_logger=retry_logger,
            )
        except Exception as exc:
            trace_event(
                active_logger,
                "agentic_openai_request_error",
                request_index=request_index,
                elapsed_seconds=round(monotonic() - started, 3),
                retry_count=retry_logger.retry_count,
                error_type=type(exc).__name__,
                status_code=getattr(exc, "status_code", None),
            )
            raise

        output = getattr(response, "output", None) or []
        function_calls = [
            item for item in output if getattr(item, "type", None) == "function_call"
        ]
        trace_event(
            active_logger,
            "agentic_openai_request_complete",
            request_index=request_index,
            elapsed_seconds=round(monotonic() - started, 3),
            retry_count=retry_logger.retry_count,
            request_id=getattr(response, "_request_id", None),
            output_items=len(output),
            function_calls=len(function_calls),
            requested_tools=[getattr(item, "name", None) for item in function_calls],
        )
        set_trace_stage(
            getattr(self, "_active_agent_state", None),
            "agent_chat",
            request_index=request_index,
            function_calls=len(function_calls),
        )
        return response


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
        agent = build_openai_agent(
            tools=step_tools_list,
            model=f"openai/{self.model}" if "/" not in self.model else self.model,
            response_model=self.response_model,
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
                resp, inputs, _ = agent.chat(
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
        logger,
        trace_path: Path,
        k: int = 10,
        format_params: dict[str, Any] | None = None,
    ) -> AgentResponse:
        inputs = [{"role": "system", "content": self.system_prompt}]
        agent_state = {"num_tool_calls": 0}
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
            )
        logger.info("agentic_output %s", resp.output_parsed if resp else None)
        if resp and hasattr(resp.output_parsed, "ranked_results"):
            ranked_results = resp.output_parsed.ranked_results or []
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
        if output and hasattr(output, "ranked_results"):
            output = list(output.ranked_results or [])[:k]
        return AgentResponse(
            output=output,
            trace_path=trace_path,
            num_tool_calls=num_tool_calls,
        )
