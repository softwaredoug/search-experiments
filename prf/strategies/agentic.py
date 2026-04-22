from __future__ import annotations

from cheat_at_search.strategy import SearchStrategy

from prf.agentic import DEFAULT_SYSTEM_PROMPT, SearchResultsIds, search
from prf.mapping import build_doc_id_lookup, doc_ids_to_indices
from prf.tools import build_search_tools


class AgenticSearchStrategy(SearchStrategy):
    _type = "agentic"

    def __init__(
        self,
        corpus,
        workers: int = 1,
        model: str = "gpt-5-mini",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list[str] | None = None,
        tools=None,
    ):
        if tools is not None:
            self.tools = tools
        else:
            tool_names = search_tools or ["bm25"]
            self.tools = build_search_tools(corpus, tool_names)
        self.model = model
        self.system_prompt = system_prompt
        self._lookup = build_doc_id_lookup(corpus)
        super().__init__(corpus, workers=workers)

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
