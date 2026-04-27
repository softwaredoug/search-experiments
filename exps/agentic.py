from __future__ import annotations

import logging
import textwrap
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Any, Iterable, Optional

import openai
from pydantic import BaseModel, Field
from typing_extensions import Literal


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


class SearchResultsIds(BaseModel):
    """The ranked, top 10 search results ordered most relevant to least."""

    results_summary: str = Field(
        description="The message from you summarizing what you found"
    )
    ranked_results: list[str] = Field(
        description="Top ranked search results (their doc_ids)"
    )


class SearchResult(BaseModel):
    """A search result and your best guess at relevance."""

    doc_id: int = Field(description="The doc id of the search result")
    grade: Literal["☹️", "😑", "😃"] = Field(
        description="How relevant this is to the query, in your estimation"
    )


class SearchResultsGraded(BaseModel):
    """The ranked, top 10 search results ordered most relevant to least."""

    results_summary: str = Field(
        description="The message from you summarizing what you found"
    )
    ranked_results: list[SearchResult] = Field(description="Ranked search results")


@dataclass
class ToolAdapter:
    args_model: type
    tool_spec: dict
    call: callable


def make_tool_info(tools: Iterable[callable]) -> dict[str, ToolAdapter]:
    from cheat_at_search.agent.pydantize import make_tool_adapter

    tool_info: dict[str, ToolAdapter] = {}
    for tool in tools:
        args_model, tool_spec, call = make_tool_adapter(tool)
        tool_info[tool.__name__] = ToolAdapter(args_model, tool_spec, call)
    return tool_info


def call_tool(tool_info: dict[str, ToolAdapter], item, agent_state, logger=None) -> dict:
    if logger is None:
        logger = logging.getLogger(__name__)
    if agent_state is not None:
        agent_state["num_tool_calls"] = agent_state.get("num_tool_calls", 0) + 1
    tool_name = item.name
    tool = tool_info[tool_name]
    tool_args = tool.args_model.model_validate_json(item.arguments)

    logger.info("Calling %s with args %s", tool_name, tool_args)
    py_resp, json_resp = tool.call(tool_args, agent_state=agent_state)
    logger.info("output %s", py_resp)
    return {
        "type": "function_call_output",
        "call_id": item.call_id,
        "output": json_resp,
    }


def stop_iterations(
    inputs: list,
    num_loops: int,
    *,
    iterations: int,
    agent_state: Optional[dict],
) -> bool:
    return num_loops >= int(iterations)


def stop_tool_calls(
    inputs: list,
    num_loops: int,
    *,
    tool_calls: int,
    agent_state: Optional[dict],
) -> bool:
    if agent_state is None:
        return False
    return int(agent_state.get("num_tool_calls", 0)) >= int(tool_calls)


STOPPERS = {
    "iterations": stop_iterations,
    "tool_calls": stop_tool_calls,
}


def agent_run(
    tool_info: dict[str, ToolAdapter],
    text_format,
    inputs,
    model: str = "gpt-5-nano",
    agent_state: Optional[dict] = None,
    summary: bool = True,
    logger=None,
):
    if logger is None:
        logger = logging.getLogger(__name__)
    tool_calls = True
    resp = None
    while tool_calls:
        failing = True
        num_failures = 0
        while failing:
            try:
                resp = openai.responses.parse(
                    model=model,
                    input=inputs,
                    tools=[tool.tool_spec for tool in tool_info.values()],
                    reasoning={
                        "effort": "medium",
                        "summary": "auto" if summary else "none",
                    },
                    text_format=text_format,
                )
                failing = False
            except Exception:
                failing = True
                num_failures += 1
                if num_failures > 3:
                    raise
                sleep(1)
        inputs += resp.output
        if summary:
            usage = resp.usage
            logger.info("--")
            logger.info("InpTok: %s", usage.input_tokens)
            logger.info("OutTok: %s", usage.output_tokens)
            for item in resp.output:
                if item.type == "reasoning":
                    logger.info("Reasoning:")
                    for summary_item in item.summary:
                        logger.info("%s\n", textwrap.fill(summary_item.text, 80))
                    item.summary = []

        for item in resp.output:
            tool_calls = False
            if item.type == "function_call":
                tool_calls = True
                tool_response = call_tool(
                    tool_info, item, agent_state=agent_state, logger=logger
                )
                inputs.append(tool_response)
    return resp, inputs


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
        if name not in STOPPERS:
            raise ValueError(f"Unknown stop condition: {name}")
        stops.append({"name": name, "params": params})
    return stops


def normalize_stops_for_cache(stop_config: list | None) -> list[dict[str, Any]]:
    return normalize_stops(stop_config)


def search(
    tools: Optional[list[callable]] = None,
    inputs: Optional[list[dict]] = None,
    agent_state: Optional[dict] = None,
    model: str = "gpt-5",
    text_format=SearchResultsIds,
    logger=None,
    stop: list | None = None,
    reprompt: str | None = None,
):
    resp = None
    if tools is None:
        tools = []
    if inputs is None:
        inputs = []
    if agent_state is None:
        agent_state = {}
    if reprompt is not None and not isinstance(reprompt, str):
        raise ValueError("reprompt must be a string when provided.")
    tool_info = make_tool_info(tools)
    stops = normalize_stops(stop)
    if not stops:
        resp, _ = agent_run(
            tool_info,
            text_format=text_format,
            inputs=inputs,
            model=model,
            agent_state=agent_state,
            logger=logger,
        )
        return resp.output_parsed
    num_loops = 0
    while True:
        resp, inputs = agent_run(
            tool_info,
            text_format=text_format,
            inputs=inputs,
            model=model,
            agent_state=agent_state,
            logger=logger,
        )
        num_loops += 1
        if any(
            STOPPERS[stopper["name"]](
                inputs,
                num_loops,
                agent_state=agent_state,
                **stopper["params"],
            )
            for stopper in stops
        ):
            break
        if reprompt:
            inputs.append({"role": "user", "content": reprompt})
    return resp.output_parsed


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

def _grade_to_emoji(grade):
    if grade == 0:
        return "☹️"
    if grade == 1:
        return "😑"
    if grade == 2:
        return "😃"
    return "☹️"


def grades(query: str, search_results: SearchResultsGraded):
    from cheat_at_search.wands_data import labeled_query_products

    query_judgments = labeled_query_products[labeled_query_products["query"] == query]
    results = []
    for search_result in search_results.ranked_results:
        doc_id = search_result.doc_id
        doc_judgments = query_judgments[query_judgments["doc_id"] == doc_id]
        if len(doc_judgments) == 0:
            results.append((doc_id, _grade_to_emoji(None)))
        else:
            grade = int(doc_judgments["grade"].values[0])
            results.append((doc_id, _grade_to_emoji(grade)))
    return results


def count_smileys(gradeds):
    count = 0
    for graded in gradeds:
        if graded[1] == "😃":
            count += 1
    return count


def degrade_hook_check(query: str):
    def search_degrade_hook(resp, inputs):
        all_graded = []
        for input_item in inputs:
            if hasattr(input_item, "content") and input_item.content is not None:
                content = input_item.content
                if content and hasattr(content[-1], "parsed"):
                    result = content[-1].parsed
                    if isinstance(result, SearchResultsGraded):
                        all_graded.append(grades(query, result))
        if len(all_graded) > 1:
            last_smileys = count_smileys(all_graded[-2])
            current_smileys = count_smileys(all_graded[-1])
            if last_smileys > current_smileys:
                inputs.append(
                    {
                        "role": "user",
                        "content": (
                            "Oh this isn't good, it turns out: You've degraded your relevance, "
                            f"previously found {last_smileys} relevant results , and now found "
                            f"{current_smileys}. Please try again"
                        ),
                    }
                )
                return False
        return True

    return search_degrade_hook
