from __future__ import annotations

import hashlib
import json

from cheat_at_search.strategy import SearchStrategy

from exps.agentic import DEFAULT_SYSTEM_PROMPT, SearchResultsIds, search
from exps.mapping import build_doc_id_lookup, doc_ids_to_indices
from exps.tools import build_search_tools


class AgenticSearchStrategy(SearchStrategy):
    _type = "agentic"

    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list[str] | None = None,
        embeddings_device: str | None = None,
    ):
        self.embeddings_device = embeddings_device
        tool_names = search_tools or ["bm25"]
        self.search_tools = list(tool_names)
        self.tools = build_search_tools(
            corpus, tool_names, embeddings_device=embeddings_device
        )
        self.model = model
        self.system_prompt = system_prompt
        self._lookup = build_doc_id_lookup(corpus)
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
        if device and "embeddings_device" not in build_params:
            tool_names = build_params.get("search_tools")
            if tool_names is None or "embeddings" in tool_names:
                build_params["embeddings_device"] = device
        return cls(corpus, workers=workers, **build_params)

    def search(self, query: str, k: int = 10):
        agentic_query = "Find me: " + query
        inputs = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": agentic_query},
        ]
        resp = search(
            tools=self.tools,
            inputs=inputs,
            model=self.model,
            text_format=SearchResultsIds,
        )
        ranked_results = resp.ranked_results[:k]
        if self._lookup:
            ranked_results = doc_ids_to_indices(ranked_results, self._lookup)
        return ranked_results, [1.0] * len(ranked_results)

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "search_tools": self.search_tools,
            "embeddings_device": self.embeddings_device,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
