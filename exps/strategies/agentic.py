from __future__ import annotations

import hashlib
import json

from cheat_at_search.strategy import SearchStrategy

from exps.agentic import (
    DEFAULT_SYSTEM_PROMPT,
    SearchResultsIds,
    normalize_stops_for_cache,
    search,
    trace_logger,
)
from exps.mapping import build_doc_id_lookup, doc_ids_to_indices
from exps.paths import AGENTIC_TRACE_ROOT
from exps.tools import (
    build_search_tools,
    normalize_search_tools,
    normalize_search_tools_for_cache,
)


class AgenticSearchStrategy(SearchStrategy):
    _type = "agentic"

    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list | None = None,
        stop: list | None = None,
        reprompt: str | None = None,
        dataset: str | None = None,
        embeddings_device: str | None = None,
    ):
        self.embeddings_device = embeddings_device
        self.dataset = dataset
        tool_config = search_tools or ["bm25"]
        self.search_tools = tool_config
        self.stop = stop
        self.reprompt = reprompt
        self.tools = build_search_tools(
            corpus,
            tool_config,
            embeddings_device=embeddings_device,
            dataset_name=dataset,
        )
        self.model = model
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
        dataset: str | None = None,
        **kwargs,
    ):
        build_params = dict(params)
        if device and "embeddings_device" not in build_params:
            tool_config = build_params.get("search_tools") or ["bm25"]
            tool_names = [tool["name"] for tool in normalize_search_tools(tool_config)]
            if "embeddings" in tool_names:
                build_params["embeddings_device"] = device
        strategy = cls(corpus, workers=workers, dataset=dataset, **build_params)
        strategy.dataset = dataset
        return strategy

    def search(self, query: str, k: int = 10):
        dataset = getattr(self, "dataset", None) or "unknown"
        agentic_query = "Find me: " + query
        inputs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": agentic_query},
        ]
        agent_state = {"num_tool_calls": 0}
        ranked_results: list[int] = []
        tries = 0
        while len(ranked_results) < k and tries < 3:
            with trace_logger(AGENTIC_TRACE_ROOT, dataset, query) as (logger, trace_path):
                logger.info("Query: %s", query)
                resp = search(
                    tools=self.tools,
                    inputs=inputs,
                    agent_state=agent_state,
                    model=self.model,
                    text_format=SearchResultsIds,
                    logger=logger,
                    stop=self.stop,
                    reprompt=self.reprompt,
                )
                ranked_results = resp.ranked_results[:k]
                if self._lookup:
                    ranked_results = doc_ids_to_indices(ranked_results, self._lookup)
            self.traces[query] = str(trace_path)
            self.num_tool_calls[query] = int(agent_state.get("num_tool_calls", 0))
            tries += 1
        if len(ranked_results) != k:
            raise ValueError(
                "Agentic search must return k results. "
                f"Got {len(ranked_results)} of {k} for query '{query}' after {tries} tries."
            )
        return ranked_results, [1.0] * len(ranked_results)

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "search_tools": normalize_search_tools_for_cache(self.search_tools),
            "stop": normalize_stops_for_cache(self.stop),
            "reprompt": self.reprompt,
            "embeddings_device": self.embeddings_device,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
