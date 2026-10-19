from __future__ import annotations

import os
from typing import Annotated, Union

import numpy as np

from cheat_at_search.embeddings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    load_model,
    load_or_create_embeddings,
)
from cheat_at_search.passage_fn import make_passage_fn
from exps.tools.results import format_corpus_result


def make_embedding_tool(
    corpus,
    device: str | None = None,
    *,
    model_name: str | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
):
    model_name = model_name or DEFAULT_MODEL_NAME

    passage_fn = make_passage_fn(document_prefix)

    show_progress = os.getenv("EXPS_EMBEDDING_PROGRESS", "1") != "0"
    embeddings, model = load_or_create_embeddings(
        corpus,
        passage_fn=passage_fn,
        model_name=model_name,
        device=device,
        chunk_size=DEFAULT_CHUNK_SIZE,
        show_progress=show_progress,
    )
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(list(embeddings))
    if model is None:
        model = load_model(model_name, device=device)

    def search_embeddings(
        query: Annotated[str, "Natural-language search query describing the desired document."],
        limit: Annotated[int, "Maximum number of ranked results to return; at most 100."] = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search a corpus using embeddings over title/description.

        Args:
            query: The search query string - natural language query
            limit: The number of top results to return (max 100).

        This is an embedding search over concatenated title + description.
        """
        if limit > 100:
            raise ValueError("limit must be <= 100")
        if query_prefix:
            query = f"{query_prefix}{query}"
        query_embedded = model.encode(query)
        similarity_scores = np.dot(embeddings, query_embedded)

        top_k_indices = np.argsort(similarity_scores)[-limit:][::-1]
        similarity_scores = similarity_scores[top_k_indices]
        top_rows = corpus.iloc[top_k_indices].copy()
        top_rows.loc[:, "score"] = similarity_scores

        results = []
        for _, row in top_rows.iterrows():
            results.append(format_corpus_result(row, score=row.get("score", 0.0), fallback_id=row.name))
        return results

    return search_embeddings
