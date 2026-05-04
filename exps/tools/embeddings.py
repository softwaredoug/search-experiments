from __future__ import annotations

from typing import Union

import numpy as np

from cheat_at_search.embeddings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    load_model,
    load_or_create_embeddings,
)
from exps.embeddings_utils import make_passage_fn


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

    embeddings, model = load_or_create_embeddings(
        corpus,
        passage_fn=passage_fn,
        model_name=model_name,
        device=device,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )
    if model is None:
        model = load_model(model_name, device=device)

    def search_embeddings(
        question: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search a corpus using embeddings over title/description.

        Args:
            question: The search query string - natural language query
            top_k: The number of top results to return (max 20).

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
