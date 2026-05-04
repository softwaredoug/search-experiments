from __future__ import annotations

import threading

import numpy as np

from cheat_at_search.embeddings import DEFAULT_MODEL_NAME, load_model
from cheat_at_search.tokenizers import snowball_tokenizer


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
    model = _minilm_guard_model()
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


def _minilm_guard_model():
    global _MINILM_GUARD_MODEL
    if _MINILM_GUARD_MODEL is None:
        with _MINILM_GUARD_LOCK:
            if _MINILM_GUARD_MODEL is None:
                _MINILM_GUARD_MODEL = load_model(DEFAULT_MODEL_NAME)
    return _MINILM_GUARD_MODEL


_MINILM_GUARD_MODEL = None
_MINILM_GUARD_LOCK = threading.Lock()
