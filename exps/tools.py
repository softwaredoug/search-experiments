from __future__ import annotations

from typing import Optional, Union

import numpy as np
from searcharray import SearchArray

from cheat_at_search.tokenizers import snowball_tokenizer
from exps.embeddings import _minilm_model, load_or_create_embeddings


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
        print("search", keywords)
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


def make_embedding_tool(corpus, device: str | None = None):
    embeddings = load_or_create_embeddings(corpus, device=device)

    def search_embeddings(
        keywords: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search a corpus using MiniLM embeddings over title/description.

        This is an embedding search over concatenated title + description.
        """
        model = _minilm_model(device=device)
        query_embedded = model.encode(keywords)
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


def make_guarded_bm25_tool(tool_fn: callable, func_name: Optional[str] = None):
    name = func_name or tool_fn.__name__

    def guarded(
        keywords: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]] | str:
        if not agent_state:
            print(f"Calling {name} {keywords} -- {top_k}")
            return tool_fn(keywords, top_k)

        q_guard = agent_state["past_queries"].get(name)
        if q_guard is None:
            q_guard = []
            agent_state["past_queries"][name] = q_guard

        if keywords in q_guard:
            err_msg = (
                "Error! You've already tried query: "
                + keywords
                + " Be more creative and explore more!"
            )
            print(name, err_msg)
            return err_msg

        print(f"Calling {name} {keywords} -- {top_k}")
        q_guard.append(keywords)
        agent_state["past_queries"][name] = q_guard
        return tool_fn(keywords, top_k)

    guarded.__name__ = name
    return guarded


def make_guarded_embedding_tool(tool_fn: callable, func_name: Optional[str] = None):
    name = func_name or tool_fn.__name__

    def guarded(
        keywords: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]] | str:
        if not agent_state:
            print(f"Calling {name} {keywords} -- {top_k}")
            return tool_fn(keywords, top_k)

        q_guard = agent_state["past_queries"].get(name)
        if q_guard is None:
            q_guard = []
            agent_state["past_queries"][name] = q_guard

        if keywords in q_guard:
            err_msg = (
                "Error! You've already tried query: "
                + keywords
                + " Be more creative and explore more!"
            )
            print(name, err_msg)
            return err_msg

        print(f"Calling {name} {keywords} -- {top_k}")
        q_guard.append(keywords)
        agent_state["past_queries"][name] = q_guard
        return tool_fn(keywords, top_k)

    guarded.__name__ = name
    return guarded


TOOL_BUILDERS = {
    "bm25": make_bm25_tool,
    "embeddings": make_embedding_tool,
}


def build_search_tools(
    corpus,
    tool_names: list[str],
    embeddings_device: str | None = None,
):
    tools = []
    for tool_name in tool_names:
        builder = TOOL_BUILDERS.get(tool_name)
        if builder is None:
            raise ValueError(f"Unknown search tool: {tool_name}")
        if tool_name == "embeddings":
            tools.append(builder(corpus, device=embeddings_device))
        else:
            tools.append(builder(corpus))
    return tools
