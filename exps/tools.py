from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
from searcharray import SearchArray

from cheat_at_search.tokenizers import snowball_tokenizer
from exps.embeddings import DEFAULT_MODEL_NAME, _minilm_model, load_or_create_embeddings


def _ensure_snowball_field(corpus, field: str) -> None:
    snowball_field = f"{field}_snowball"
    if snowball_field in corpus or field not in corpus:
        return
    corpus[snowball_field] = SearchArray.index(corpus[field], snowball_tokenizer)


def make_bm25_tool(corpus, title_boost: float = 10.0, description_boost: float = 1.0):
    _ensure_snowball_field(corpus, "title")
    _ensure_snowball_field(corpus, "description")

    def search_bm25(
        keywords: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search a corpus using BM25 over title/description fields.

        Args:
            keywords: The search query string.
            top_k: The number of top results to return.

        Returns:
            Search results as a list of dictionaries with 'id', 'title',
            'description', and 'score' keys.
        """
        bm25_scores = np.zeros(len(corpus))
        for term in snowball_tokenizer(keywords):
            bm25_scores += corpus["title_snowball"].array.score(term) * title_boost
            bm25_scores += (
                corpus["description_snowball"].array.score(term) * description_boost
            )

        top_k_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        bm25_scores = bm25_scores[top_k_indices]
        top_rows = corpus.iloc[top_k_indices].copy()
        top_rows.loc[:, "score"] = bm25_scores

        results = []
        for _, row in top_rows.iterrows():
            results.append(
                {
                    "id": row.get("doc_id", row.name),
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "score": row.get("score", 0.0),
                }
            )
        return results

    return search_bm25


def make_embedding_tool(
    corpus,
    device: str | None = None,
    dataset_name: str | None = None,
    *,
    model_name: str | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
):
    model_name = model_name or DEFAULT_MODEL_NAME
    embeddings = load_or_create_embeddings(
        corpus,
        model_name=model_name,
        device=device,
        dataset_name=dataset_name,
        document_prefix=document_prefix,
    )
    model = _minilm_model(model_name, device=device)

    def search_embeddings(
        question: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search a corpus using embeddings over title/description.

        Args:
            question: The search query string - natural language query
            top_k: The number of top results to return.

        This is an embedding search over concatenated title + description.
        """
        if query_prefix:
            question = f"{query_prefix}{question}"
        query_embedded = model.encode(question)
        similarity_scores = np.dot(embeddings, query_embedded)

        top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        similarity_scores = similarity_scores[top_k_indices]
        top_rows = corpus.iloc[top_k_indices].copy()
        top_rows.loc[:, "score"] = similarity_scores

        results = []
        for _, row in top_rows.iterrows():
            results.append(
                {
                    "id": row.get("doc_id", row.name),
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "score": row.get("score", 0.0),
                }
            )
        return results

    return search_embeddings


def guard_disallow_repeated_queries(params: dict, agent_state: dict | None) -> str | None:
    """Reject repeated queries per tool using agent_state["past_queries"]."""
    if agent_state is None:
        return None
    tool_name = params.get("tool_name")
    query = params.get("query")
    if not tool_name or query is None:
        return None
    q_guard = agent_state.setdefault("past_queries", {}).get(tool_name)
    if q_guard is None:
        q_guard = []
        agent_state["past_queries"][tool_name] = q_guard
    if query in q_guard:
        return (
            "Error! You've already tried query: "
            + query
            + " Be more creative and explore more!"
        )
    q_guard.append(query)
    agent_state["past_queries"][tool_name] = q_guard
    return None


def guard_query_min_length(
    params: dict, agent_state: dict | None, *, min_terms: int
) -> str | None:
    """Reject queries with fewer than min_terms tokens."""
    query = params.get("query", "")
    term_count = len(snowball_tokenizer(query))
    if term_count < min_terms:
        return (
            "Embedding search works best with natural language questions. "
            f"Please provide a more detailed query with at least {min_terms} terms."
        )
    return None


def guard_disallow_similar_queries(
    params: dict,
    agent_state: dict | None,
    *,
    threshold: float = 0.9,
) -> str | None:
    """Reject queries whose embeddings are too similar to prior queries."""
    if agent_state is None:
        return None
    tool_name = params.get("tool_name")
    query = params.get("query")
    if not tool_name or query is None:
        return None

    past_queries = agent_state.setdefault("past_queries", {}).get(tool_name)
    if past_queries is None:
        past_queries = []
        agent_state["past_queries"][tool_name] = past_queries

    past_embeddings = agent_state.setdefault("past_query_embeddings", {}).get(tool_name)
    model = _minilm_model(DEFAULT_MODEL_NAME)
    if past_embeddings is None and past_queries:
        past_embeddings = list(model.encode(past_queries))
        agent_state["past_query_embeddings"][tool_name] = past_embeddings

    if past_embeddings:
        query_embedding = np.asarray(model.encode(query))
        query_norm = float(np.linalg.norm(query_embedding))
        if query_norm > 0:
            for past_embedding in past_embeddings:
                past_embedding = np.asarray(past_embedding)
                past_norm = float(np.linalg.norm(past_embedding))
                if past_norm == 0:
                    continue
                similarity = float(np.dot(query_embedding, past_embedding) / (query_norm * past_norm))
                if similarity > threshold:
                    return (
                        "Error! You've already tried a very similar query. "
                        "Be more creative and explore more!"
                    )

    query_embedding = np.asarray(model.encode(query))
    past_queries.append(query)
    agent_state["past_queries"][tool_name] = past_queries
    agent_state.setdefault("past_query_embeddings", {}).setdefault(tool_name, []).append(query_embedding)
    return None


GUARDS = {
    "disallow_repeated_queries": guard_disallow_repeated_queries,
    "query_min_length": guard_query_min_length,
    "disallow_similar_queries": guard_disallow_similar_queries,
}


def _parse_guard_entry(entry: Any) -> tuple[str, dict]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and len(entry) == 1:
        name = next(iter(entry))
        params = entry[name]
        if params is None:
            params = {}
        if not isinstance(params, dict):
            if name == "disallow_similar_queries":
                return name, {"threshold": params}
            raise ValueError(f"Guard params for {name} must be a mapping.")
        return name, params
    raise ValueError("Guard entry must be a string or single-key mapping.")


def _guard_doc(guard: dict) -> str:
    guard_name = guard["name"]
    guard_params = guard.get("params", {})
    guard_fn = GUARDS.get(guard_name)
    description = ""
    if guard_fn and guard_fn.__doc__:
        description = guard_fn.__doc__.strip()
    if guard_params:
        params_str = ", ".join(f"{key}={value}" for key, value in guard_params.items())
        if description:
            return f"{guard_name}({params_str}): {description}"
        return f"{guard_name}({params_str})"
    if description:
        return f"{guard_name}: {description}"
    return guard_name


def normalize_search_tools(tool_config: list) -> list[dict[str, Any]]:
    normalized = []
    for item in tool_config:
        if isinstance(item, str):
            normalized.append({"name": item, "guards": []})
            continue
        if isinstance(item, dict) and len(item) == 1:
            tool_name = next(iter(item))
            tool_info = item[tool_name] or {}
            if not isinstance(tool_info, dict):
                raise ValueError(f"Tool config for {tool_name} must be a mapping.")
            guards_raw = tool_info.get("guards") or []
            if not isinstance(guards_raw, list):
                raise ValueError(f"Guards for {tool_name} must be a list.")
            guards = []
            for guard_entry in guards_raw:
                guard_name, guard_params = _parse_guard_entry(guard_entry)
                guards.append({"name": guard_name, "params": guard_params})
            normalized.append({"name": tool_name, "guards": guards})
            continue
        raise ValueError("search_tools entries must be strings or single-key mappings.")
    return normalized


def normalize_search_tools_for_cache(tool_config: list) -> list[dict[str, Any]]:
    normalized = normalize_search_tools(tool_config)
    normalized_sorted = []
    for tool in sorted(normalized, key=lambda item: item["name"]):
        guards = sorted(tool["guards"], key=lambda item: item["name"])
        guards = [
            {"name": guard["name"], "params": dict(sorted(guard["params"].items()))}
            for guard in guards
        ]
        normalized_sorted.append({"name": tool["name"], "guards": guards})
    return normalized_sorted


def make_guarded_search_tool(
    tool_fn: callable,
    *,
    guards: list[dict[str, Any]] | None = None,
    func_name: Optional[str] = None,
):
    name = func_name or tool_fn.__name__
    guards = guards or []

    def guarded(
        query: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]] | str:
        """Search tool wrapper that enforces configured guard checks."""
        params = {
            "tool_name": name,
            "query": query,
            "top_k": top_k,
        }
        for guard in guards:
            guard_name = guard["name"]
            guard_params = guard.get("params", {})
            guard_fn = GUARDS.get(guard_name)
            if guard_fn is None:
                raise ValueError(f"Unknown guard: {guard_name}")
            err = guard_fn(params, agent_state, **guard_params)
            if isinstance(err, str) and err:
                return err
        return tool_fn(query, top_k, agent_state)

    guarded.__name__ = name
    guarded.__doc__ = tool_fn.__doc__
    return guarded


TOOL_BUILDERS = {
    "bm25": make_bm25_tool,
    "minilm": make_embedding_tool,
    "embeddings": make_embedding_tool,
    "e5_base_v2": lambda corpus, device=None, dataset_name=None: make_embedding_tool(
        corpus,
        device=device,
        dataset_name=dataset_name,
        model_name="intfloat/e5-base-v2",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
}


def build_search_tools(
    corpus,
    tool_config: list,
    embeddings_device: str | None = None,
    dataset_name: str | None = None,
):
    tools = []
    for tool in normalize_search_tools(tool_config):
        tool_name = tool["name"]
        builder = TOOL_BUILDERS.get(tool_name)
        if builder is None:
            raise ValueError(f"Unknown search tool: {tool_name}")
        if tool_name in {"embeddings", "minilm", "e5_base_v2"}:
            tool_fn = builder(
                corpus, device=embeddings_device, dataset_name=dataset_name
            )
        else:
            tool_fn = builder(corpus)
        if tool["guards"]:
            guard_lines = ["Guards:"]
            guard_lines.extend(f"- {_guard_doc(guard)}" for guard in tool["guards"])
            guard_block = "\n".join(guard_lines)
            base_doc = tool_fn.__doc__ or ""
            tool_fn.__doc__ = f"{base_doc}\n\n{guard_block}" if base_doc else guard_block
        if tool["guards"]:
            tool_fn = make_guarded_search_tool(
                tool_fn,
                guards=tool["guards"],
                func_name=f"search_{tool_name}",
            )
        tools.append(tool_fn)
    return tools
