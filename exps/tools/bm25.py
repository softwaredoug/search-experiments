from __future__ import annotations

from typing import Union

import numpy as np
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity
from typing_extensions import Literal

from cheat_at_search.tokenizers import snowball_tokenizer


def make_bm25_tool(
    corpus,
    title_boost: float = 10.0,
    description_boost: float = 1.0,
    k1: float | None = None,
    b: float | None = None,
):
    def search_bm25(
        keywords: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search a corpus using BM25 over title/description fields.

        Args:
            keywords: The search query string.
            top_k: The number of top results to return (max 20).

        Returns:
            Search results as a list of dictionaries with 'id', 'title',
            'description', and 'score' keys.
        """
        bm25_scores = np.zeros(len(corpus))
        similarity = None
        if k1 is not None or b is not None:
            similarity = bm25_similarity(k1=k1 or 1.2, b=b or 0.75)
        for term in snowball_tokenizer(keywords):
            if similarity is None:
                bm25_scores += corpus["title_snowball"].array.score(term) * title_boost
                bm25_scores += (
                    corpus["description_snowball"].array.score(term) * description_boost
                )
            else:
                bm25_scores += (
                    corpus["title_snowball"].array.score(term, similarity=similarity)
                    * title_boost
                )
                bm25_scores += (
                    corpus["description_snowball"].array.score(
                        term, similarity=similarity
                    )
                    * description_boost
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


def _parse_weighted_fields(fields: list[str]) -> list[tuple[str, float]]:
    parsed_fields = []
    for field_entry in fields:
        if not isinstance(field_entry, str) or "^" not in field_entry:
            raise ValueError("Fields must be strings in the form 'title^9.3'.")
        field_name, weight_str = field_entry.split("^", 1)
        field_name = field_name.strip()
        if field_name not in {"title", "description"}:
            raise ValueError("Fields must be title or description.")
        try:
            weight = float(weight_str)
        except ValueError as exc:
            raise ValueError("Field weights must be numeric.") from exc
        parsed_fields.append((field_name, weight))
    if not parsed_fields:
        raise ValueError("Fields must include at least one entry.")
    return parsed_fields


def _ensure_field_indices(corpus, field_names: list[str]) -> None:
    for field_name in field_names:
        if field_name not in corpus:
            raise ValueError(f"Corpus missing required field: {field_name}")
        snowball_name = f"{field_name}_snowball"
        if snowball_name not in corpus:
            corpus[snowball_name] = SearchArray.index(corpus[field_name], snowball_tokenizer)


def make_fielded_bm25_tool(corpus):
    def fielded_bm25(
        keywords: str,
        fields: list[str],
        operator: Literal["and", "or"] = "or",
        top_k: int = 5,
        k1: float = 1.2,
        b: float = 0.75,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search a corpus using BM25 over weighted fields.

        Args:
            keywords: The search query string.
            fields: Weighted fields to search, e.g. ["title^9.3", "description^4.1"].
            operator: How to combine search terms: and/or. AND requires every term
                to appear in at least one field.
            top_k: The number of top results to return.
            k1: BM25 k1 parameter.
            b: BM25 b parameter.

        Returns:
            Search results as a list of dictionaries with 'id', 'title',
            'description', and 'score' keys.
        """
        if not isinstance(fields, list):
            raise ValueError("fields must be a list of weighted field strings.")
        parsed_fields = _parse_weighted_fields(fields)
        _ensure_field_indices(corpus, [field for field, _ in parsed_fields])

        query_tokens = snowball_tokenizer(keywords)
        scores = np.zeros(len(corpus))
        similarity = bm25_similarity(k1=k1, b=b)
        require_mask = None

        for token in query_tokens:
            field_scores = []
            for field_name, weight in parsed_fields:
                snowball_name = f"{field_name}_snowball"
                term_match = corpus[snowball_name].array.score(token, similarity=similarity)
                field_scores.append(term_match * weight)
            term_scores = sum(field_scores) if field_scores else 0
            scores += term_scores
            if operator == "and":
                term_present = sum(field_scores) > 0
                require_mask = term_present if require_mask is None else (require_mask & term_present)

        if operator == "and":
            if require_mask is not None:
                scores = scores * require_mask
        elif operator != "or":
            raise ValueError("operator must be 'and' or 'or'")

        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        scores = scores[top_k_indices]
        top_rows = corpus.iloc[top_k_indices].copy()
        top_rows.loc[:, "score"] = scores

        results = []
        for _, row in top_rows.iterrows():
            results.append(
                {
                    "id": row.get("doc_id"),
                    "title": row.get("title", ""),
                    "description": row.get("description", ""),
                    "score": row.get("score", 0.0),
                }
            )
        return results

    return fielded_bm25
