from __future__ import annotations

import hashlib
import inspect
import json

from pydantic import BaseModel, Field

from cheat_at_search.strategy import SearchStrategy
from exps.agentic.agent import build_openai_agent
from exps.mapping import build_doc_id_lookup
from exps.tools import build_search_tools, normalize_search_tools


DEFAULT_SYSTEM_PROMPT = """
Given the user's search request, generate one concise query for the search tool.
Return only a query that will help retrieve relevant documents.
""".strip()


class SearchQuery(BaseModel):
    query: str = Field(description="The query to send to the retrieval tool")


class RagSearchStrategy(SearchStrategy):
    """Generate one query with an LLM, then run one retrieval tool."""

    _type = "rag"

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
            build_params["embeddings_device"] = device
        return cls(corpus, workers=workers, dataset_name=dataset, **build_params)

    def __init__(
        self,
        corpus,
        *,
        model: str = "gpt-5-mini",
        reasoning: str = "medium",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        search_tools: list | None = None,
        workers: int = 1,
        embeddings_device: str | None = None,
        dataset_name: str | None = None,
    ):
        super().__init__(corpus, workers=workers)
        configured_tools = search_tools or ["bm25"]
        if len(configured_tools) != 1:
            raise ValueError("RAG strategies require exactly one search tool.")
        self.corpus = corpus
        self.model = model
        self.reasoning = reasoning
        self.system_prompt = system_prompt
        self.search_tools = configured_tools
        self.embeddings_device = embeddings_device
        self.dataset_name = dataset_name
        self._lookup = build_doc_id_lookup(corpus)
        self._tool = build_search_tools(
            corpus,
            configured_tools,
            embeddings_device=embeddings_device,
            dataset_name=dataset_name,
            system_prompt=system_prompt,
        )[0]

    def _generate_query(self, query: str) -> str:
        agent = build_openai_agent(
            tools=[],
            model=f"openai/{self.model}" if "/" not in self.model else self.model,
            response_model=SearchQuery,
            reasoning_level=self.reasoning,
        )
        response, _, _ = agent.chat(
            inputs=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query},
            ],
            agent_state={},
        )
        parsed = response.output_parsed
        generated_query = parsed.query if isinstance(parsed, SearchQuery) else ""
        generated_query = generated_query.strip()
        if not generated_query:
            raise ValueError("RAG query generator returned an empty query.")
        return generated_query

    def search(self, query: str, k: int = 10):
        generated_query = self._generate_query(query)
        parameters = inspect.signature(self._tool).parameters
        query_name = next(
            (name for name in ("query", "keywords", "question") if name in parameters),
            None,
        )
        if query_name is None:
            raise ValueError("RAG search tool must accept a query argument.")
        results = self._tool(
            **{query_name: generated_query}, top_k=k, agent_state={}
        )
        if isinstance(results, str):
            raise ValueError(f"RAG search tool failed: {results}")
        if not isinstance(results, list):
            raise ValueError("RAG search tool must return a list of results.")
        valid_results = [
            result
            for result in results
            if isinstance(result, dict) and result.get("id") is not None
        ]
        mapped_results = [
            (self._lookup[str(result["id"])], result)
            for result in valid_results
            if str(result["id"]) in self._lookup
        ]
        indices = [index for index, _ in mapped_results]
        scores = [float(result.get("score", 1.0)) for _, result in mapped_results]
        return indices, scores

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "reasoning": self.reasoning,
            "system_prompt": self.system_prompt,
            "search_tools": normalize_search_tools(self.search_tools),
            "embeddings_device": self.embeddings_device,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
