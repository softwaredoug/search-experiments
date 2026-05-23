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
from typing_extensions import Literal

from exps.mapping import build_doc_id_lookup, doc_ids_to_indices
from exps.run_dirs import dataset_from_trace_path, slugify
from exps.agentic.examples import append_few_shot_examples
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


class SearchResultsIds(BaseModel):
    """The ranked, top 10 search result DOC IDs ordered most relevant to least."""

    results_summary: str = Field(
        description="The message from you summarizing what you found"
    )
    next_plan: str = Field(
        description=(
            "In the form of instructions"
            "Instruct an LLM how to improve the search results beyond what you found using"
            "the tools available"
        )
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



def search(
    tools: list[callable] | None = None,
    inputs: list[dict] | None = None,
    agent_state: Optional[dict] = None,
    model: str = "gpt-5",
    text_format=SearchResultsIds,
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
        stop: list | None = None,
        reprompt: str | None = None,
        embeddings_device: str | None = None,
        trace_path: Path | None = None,
    ):
        self.embeddings_device = embeddings_device
        self.trace_path = trace_path
        dataset_name = dataset_from_trace_path(trace_path) if trace_path else None
        tool_config = search_tools or ["bm25"]
        self.search_tools = tool_config
        self.stop = stop
        self.reprompt = reprompt
        self.tools = build_search_tools(
            corpus,
            tool_config,
            embeddings_device=embeddings_device,
            dataset_name=dataset_name,
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

    def search(self, query: str, k: int = 10):
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
        inputs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        agent_state = {"num_tool_calls": 0}
        stops = normalize_stops(self.stop)
        reprompt = self.reprompt
        if reprompt is not None and not isinstance(reprompt, str):
            raise ValueError("reprompt must be a string when provided.")
        num_loops = 0
        agent = OpenAIAgent(
            tools=self.tools,
            model=f"openai/{self.model}" if "/" not in self.model else self.model,
            response_model=SearchResultsIds,
            reasoning_level=self.reasoning,
        )
        with trace_logger(query_dir) as (logger, trace_path):
            logger.info("Query: %s", query)
            while True:
                previous_inputs = list(inputs)
                resp, inputs, _ = agent.chat(inputs=inputs, agent_state=agent_state)
                new_items = inputs[len(previous_inputs) :]
                for item in new_items:
                    logger.info("agentic_output %s", item)
                num_loops += 1
                tool_calls = agent_state.get("num_tool_calls")
                if tool_calls is None:
                    tool_calls = _tool_calls_from_inputs(inputs)
                    agent_state["num_tool_calls"] = tool_calls
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

            ranked_results = resp.output_parsed.ranked_results[:k]
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

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
            "search_tools": normalize_search_tools_for_cache(self.search_tools),
            "stop": self.stop,
            "reprompt": self.reprompt,
            "embeddings_device": self.embeddings_device,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
