from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path
from typing import Any, Optional, Union

from typing_extensions import Literal

import threading

import numpy as np

from searcharray import SearchArray
from searcharray.similarity import bm25_similarity

from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.embeddings import (
    DEFAULT_MODEL_NAME,
    DEFAULT_CHUNK_SIZE,
    load_model,
    load_or_create_embeddings,
)
from exps.embeddings_utils import make_passage_fn

WANDS_TOP_CATEGORIES = [
    "Furniture",
    "Home Improvement",
    "Décor & Pillows",
    "Outdoor",
    "Storage & Organization",
    "Lighting",
    "Rugs",
    "Bed & Bath",
    "Kitchen & Tabletop",
    "Baby & Kids",
    "School Furniture and Supplies",
    "Appliances",
    "Holiday Décor",
    "Commercial Business Furniture",
    "Pet",
    "Contractor",
    "Sale",
    "Foodservice",
    "Shop Product Type",
    "Browse By Brand",
]
WandsProductCategory = Literal[
    "Furniture",
    "Home Improvement",
    "Décor & Pillows",
    "Outdoor",
    "Storage & Organization",
    "Lighting",
    "Rugs",
    "Bed & Bath",
    "Kitchen & Tabletop",
    "Baby & Kids",
    "School Furniture and Supplies",
    "Appliances",
    "Holiday Décor",
    "Commercial Business Furniture",
    "Pet",
    "Contractor",
    "Sale",
    "Foodservice",
    "Shop Product Type",
    "Browse By Brand",
]
WANDS_CATEGORY_COL = "category"


def _build_category_index(corpus, *, category_col: str) -> dict[str, np.ndarray]:
    if category_col not in corpus.columns:
        raise ValueError(f"Missing {category_col} column for category filtering.")
    values = corpus[category_col].fillna("").astype(str)
    category_index: dict[str, list[int]] = {}
    for idx, value in enumerate(values.tolist()):
        if not value:
            continue
        category_index.setdefault(value, []).append(idx)
    return {key: np.asarray(indices, dtype=int) for key, indices in category_index.items()}


def _category_indices(
    category_index: dict[str, np.ndarray],
    product_categories: str | list[str] | None,
):
    if not product_categories:
        return None
    if isinstance(product_categories, str):
        product_categories = [product_categories]
    indices = [
        category_index.get(category)
        for category in product_categories
        if category
    ]
    indices = [idx for idx in indices if idx is not None]
    if not indices:
        return np.asarray([], dtype=int)
    return np.unique(np.concatenate(indices))


def _parse_feature_key(feature: str) -> str:
    if not feature:
        return ""
    key, *_ = feature.split(":", 1)
    return key.strip()


def _build_feature_lookup(
    corpus,
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, str]]]]:
    if "features" not in corpus.columns:
        raise ValueError("Missing features column for WANDS feature lookup.")
    doc_ids = (
        corpus["doc_id"].astype(str).tolist()
        if "doc_id" in corpus.columns
        else [str(idx) for idx in corpus.index]
    )
    feature_values: dict[str, set[str]] = {}
    doc_features: dict[str, list[tuple[str, str]]] = {}
    for doc_id, features in zip(doc_ids, corpus["features"].tolist()):
        if not features:
            continue
        for item in features:
            item_str = str(item)
            key, _, value = item_str.partition(":")
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            if value:
                feature_values.setdefault(key, set()).add(value)
            doc_features.setdefault(doc_id, []).append((key, value))
    feature_values_map = {key: sorted(values) for key, values in feature_values.items()}
    return feature_values_map, doc_features


def make_bm25_tool(corpus, title_boost: float = 10.0, description_boost: float = 1.0):
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


def make_wands_bm25_tool(
    corpus,
    title_boost: float = 10.0,
    description_boost: float = 1.0,
):
    category_index = _build_category_index(corpus, category_col=WANDS_CATEGORY_COL)

    def search_bm25_wands(
        keywords: str,
        product_categories: list[WandsProductCategory] | None = None,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search WANDS with BM25 over title/description and optional category filter.

        Args:
            keywords: The search query string.
            product_categories: Optional category filters. Categorization may be imperfect;
                consider searching both with and without categories.
            top_k: The number of top results to return (max 20).

        Returns:
            Search results as a list of dictionaries with 'id', 'title',
            'description', and 'score' keys.
        """
        print(f"B - Searching WANDS for keywords: {keywords} with categories: {product_categories}")
        indices = _category_indices(category_index, product_categories)
        if indices is None:
            working_corpus = corpus
        elif indices.size == 0:
            return []
        else:
            working_corpus = corpus.iloc[indices]

        bm25_scores = np.zeros(len(working_corpus))
        for term in snowball_tokenizer(keywords):
            bm25_scores += working_corpus["title_snowball"].array.score(term) * title_boost
            bm25_scores += (
                working_corpus["description_snowball"].array.score(term) * description_boost
            )

        top_k_indices = np.argsort(bm25_scores)[-top_k:][::-1]
        bm25_scores = bm25_scores[top_k_indices]
        top_rows = working_corpus.iloc[top_k_indices].copy()
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

    return search_bm25_wands


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


def make_wands_embedding_tool(
    corpus,
    device: str | None = None,
    *,
    model_name: str | None = None,
    query_prefix: str | None = None,
    document_prefix: str | None = None,
):
    model_name = model_name or DEFAULT_MODEL_NAME
    category_index = _build_category_index(corpus, category_col=WANDS_CATEGORY_COL)

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

    def search_embeddings_wands(
        product_description: str,
        product_categories: list[WandsProductCategory] | None = None,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search WANDS using embeddings with an optional category filter.

        Args:
            product_description: The product being looked for.
            product_categories: Optional category filters.
            top_k: The number of top results to return (max 20).

        This is an embedding search over concatenated title + description.
        """
        print(f"E - Searching WANDS for query: {product_description} with categories: {product_categories}")
        if query_prefix:
            question = f"{query_prefix}{product_description}"
        query_embedded = model.encode(question)
        query_embedded = np.asarray(query_embedded)

        indices = _category_indices(category_index, product_categories)
        if indices is None:
            working_embeddings = embeddings
            working_corpus = corpus
            source_indices = None
        elif indices.size == 0:
            return []
        else:
            working_embeddings = embeddings[indices]
            working_corpus = corpus.iloc[indices]
            source_indices = indices

        similarity_scores = np.dot(working_embeddings, query_embedded)
        top_k_indices = np.argsort(similarity_scores)[-top_k:][::-1]
        similarity_scores = similarity_scores[top_k_indices]
        if source_indices is not None:
            top_doc_indices = source_indices[top_k_indices]
            top_rows = corpus.iloc[top_doc_indices].copy()
        else:
            top_rows = working_corpus.iloc[top_k_indices].copy()
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

    return search_embeddings_wands


def make_check_features_wands_tool(corpus):
    feature_values, doc_features = _build_feature_lookup(corpus)
    feature_names = list(feature_values.keys())
    if not feature_names:
        raise ValueError("No features found for WANDS feature lookup.")
    model = load_model(DEFAULT_MODEL_NAME)
    feature_embeddings = model.encode(feature_names, convert_to_numpy=True)
    feature_embeddings = np.asarray(feature_embeddings)
    norms = np.linalg.norm(feature_embeddings, axis=1)
    norms[norms == 0] = 1.0
    feature_embeddings = feature_embeddings / norms[:, None]
    feature_index = {name: idx for idx, name in enumerate(feature_names)}

    def check_features_wands(
        doc_id: str,
        feature_names: list[str],
        agent_state=None,
    ) -> dict[str, dict[str, list[str]]]:
        """Find feature keys within a product that match feature name queries.

        Args:
            doc_id: Product doc_id to inspect.
            feature_names: Feature name queries (examples: color, wood, non-toxic,
              hydrauliclift, adjustableheight, recliningtypedetails, upholsterycolor,
              etc)

        Returns:
            Dictionary mapping queried feature to matched feature->values.
        """
        if not feature_names:
            return {}
        doc_key = str(doc_id)
        features = doc_features.get(doc_key)
        if not features:
            return {}
        doc_feature_map: dict[str, list[str]] = {}
        for key, value in features:
            doc_feature_map.setdefault(key, []).append(value)

        key_indices = [feature_index[key] for key in doc_feature_map.keys() if key in feature_index]
        if not key_indices:
            return {}
        doc_embeddings = feature_embeddings[key_indices]
        keys_list = [key for key in doc_feature_map.keys() if key in feature_index]

        results: dict[str, dict[str, list[str]]] = {}
        for feature_name in feature_names:
            if not feature_name:
                continue
            query_embedded = model.encode(feature_name, convert_to_numpy=True)
            query_embedded = np.asarray(query_embedded)
            query_norm = np.linalg.norm(query_embedded)
            if query_norm == 0:
                continue
            query_embedded = query_embedded / query_norm
            similarities = doc_embeddings @ query_embedded
            top_local = np.argsort(similarities)[-5:][::-1]
            matches: dict[str, list[str]] = {}
            for local_idx in top_local:
                score = float(similarities[local_idx])
                if score <= 0.5:
                    continue
                matched_feature = keys_list[local_idx]
                values = doc_feature_map.get(matched_feature, [])
                if not values:
                    continue
                matches.setdefault(matched_feature, [])
                for value in values:
                    if value not in matches[matched_feature]:
                        matches[matched_feature].append(value)
            if matches:
                results[feature_name] = matches
        return results

    return check_features_wands


def _find_latest_reranker_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"codegen path not found: {path}")
    round_files = list(path.glob("reranker_round_*.py"))
    if round_files:
        def round_number(file_path: Path) -> int:
            match = re.search(r"reranker_round_(\d+)\.py", file_path.name)
            return int(match.group(1)) if match else -1

        return max(round_files, key=round_number)
    fallback = path / "reranker.py"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No reranker files found in {path}")


def _load_reranker_fn(path: Path, dataset_name: str | None) -> tuple[callable, str]:
    module_name = f"codegen_reranker_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load reranker module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if dataset_name:
        fn_name = f"rerank_{dataset_name}"
        reranker_fn = getattr(module, fn_name, None)
        if reranker_fn is not None:
            return reranker_fn, fn_name
    reranker_fn = getattr(module, "rerank", None)
    if reranker_fn is None:
        raise ValueError(f"No rerank function found in {path}")
    return reranker_fn, "rerank"


def make_codegen_tool(
    corpus,
    *,
    tool_config: dict[str, Any],
    embeddings_device: str | None = None,
    dataset_name: str | None = None,
):
    path = tool_config.get("path")
    if not path:
        raise ValueError("codegen tool requires a path.")
    tool_name = tool_config.get("name") or "search"
    description = tool_config.get("description") or "Search the dataset and return results."
    return_fields = tool_config.get("return_fields") or []
    if not isinstance(return_fields, list):
        raise ValueError("return_fields must be a list when provided.")
    missing_fields = [field for field in return_fields if field not in corpus.columns]
    if missing_fields:
        missing_str = ", ".join(missing_fields)
        raise ValueError(f"return_fields not found in corpus: {missing_str}")
    dependencies = tool_config.get("dependencies") or []
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be a list of tool entries.")
    dependency_tools = build_search_tools(
        corpus,
        dependencies,
        embeddings_device=embeddings_device,
        dataset_name=dataset_name,
    )
    dependency_map = {tool.__name__: tool for tool in dependency_tools}
    reranker_path = _find_latest_reranker_path(Path(path).expanduser())
    reranker_fn, reranker_name = _load_reranker_fn(reranker_path, dataset_name)
    signature = inspect.signature(reranker_fn)
    params = list(signature.parameters.values())
    required_deps = []
    if params:
        for param in params[1:]:
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            required_deps.append(param.name)
    missing_deps = [dep for dep in required_deps if dep not in dependency_map]
    if missing_deps:
        missing_str = ", ".join(missing_deps)
        raise ValueError(f"codegen tool missing dependencies: {missing_str}")

    doc_lookup = corpus.set_index("doc_id", drop=False)

    def _to_builtin(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _coerce_doc_id(value):
        value = _to_builtin(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value

    def search(
        query: str,
        top_k: int = 10,
        agent_state=None,
        **kwargs,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search the corpus, return top results."""
        if top_k > 20:
            return "Error! top_k must be <= 20."
        reranker_results = reranker_fn(query=query, **dependency_map, **kwargs)
        results = []
        for rank, item in enumerate(reranker_results or []):
            if isinstance(item, dict):
                doc_id = _coerce_doc_id(item.get("id") or item.get("doc_id"))
                score = _to_builtin(item.get("score"))
            else:
                doc_id = _coerce_doc_id(item)
                score = None
            if doc_id is None or doc_id not in doc_lookup.index:
                continue
            row = doc_lookup.loc[doc_id]
            entry = {
                "id": int(_to_builtin(row.get("doc_id", doc_id))),
                "title": str(_to_builtin(row.get("title", ""))),
                "description": str(_to_builtin(row.get("description", ""))),
                "score": float(score) if score is not None else 1.0 / (rank + 1),
            }
            for field in return_fields:
                entry[field] = _to_builtin(row.get(field))
            results.append(entry)
            if len(results) >= top_k:
                break
        return results

    search.__name__ = tool_name
    search.__doc__ = description
    search.__codegen_reranker__ = reranker_name
    return search


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


def _normalize_tool_config(tool_info: dict) -> dict[str, Any]:
    return {key: value for key, value in tool_info.items() if key != "guards"}


def normalize_search_tools(tool_config: list) -> list[dict[str, Any]]:
    normalized = []
    for item in tool_config:
        if isinstance(item, str):
            normalized.append({"name": item, "guards": [], "config": {}})
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
            normalized.append(
                {
                    "name": tool_name,
                    "guards": guards,
                    "config": _normalize_tool_config(tool_info),
                }
            )
            continue
        raise ValueError("search_tools entries must be strings or single-key mappings.")
    return normalized


def _normalize_tool_config_for_cache(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in config.items():
        if key == "dependencies":
            if not isinstance(value, list):
                raise ValueError("dependencies must be a list of tool entries.")
            normalized[key] = normalize_search_tools_for_cache(value)
            continue
        if isinstance(value, Path):
            normalized[key] = str(value)
            continue
        normalized[key] = value
    return dict(sorted(normalized.items()))


def normalize_search_tools_for_cache(tool_config: list) -> list[dict[str, Any]]:
    normalized = normalize_search_tools(tool_config)
    normalized_sorted = []
    for tool in sorted(normalized, key=lambda item: item["name"]):
        guards = sorted(tool["guards"], key=lambda item: item["name"])
        guards = [
            {"name": guard["name"], "params": dict(sorted(guard["params"].items()))}
            for guard in guards
        ]
        config = tool.get("config") or {}
        normalized_sorted.append(
            {
                "name": tool["name"],
                "guards": guards,
                "config": _normalize_tool_config_for_cache(config),
            }
        )
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
        if top_k > 20:
            return "Error! top_k must be <= 20."
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
    "fielded_bm25": make_fielded_bm25_tool,
    "minilm": make_embedding_tool,
    "embeddings": make_embedding_tool,
    "codegen": make_codegen_tool,
    "e5_base_v2": lambda corpus, device=None: make_embedding_tool(
        corpus,
        device=device,
        model_name="intfloat/e5-base-v2",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    "bm25_wands": make_wands_bm25_tool,
    "minilm_wands": make_wands_embedding_tool,
    "e5_base_v2_wands": lambda corpus, device=None: make_wands_embedding_tool(
        corpus,
        device=device,
        model_name="intfloat/e5-base-v2",
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
    "check_features_wands": make_check_features_wands_tool,
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
        if tool_name.endswith("_wands"):
            if dataset_name != "wands":
                raise ValueError(f"{tool_name} is only available for wands dataset.")
        builder = TOOL_BUILDERS.get(tool_name)
        if builder is None:
            raise ValueError(f"Unknown search tool: {tool_name}")
        if tool_name == "codegen":
            tool_fn = builder(
                corpus,
                tool_config=tool.get("config") or {},
                embeddings_device=embeddings_device,
                dataset_name=dataset_name,
            )
        elif tool_name in {"embeddings", "minilm", "e5_base_v2"}:
            tool_fn = builder(corpus, device=embeddings_device)
        else:
            tool_fn = builder(corpus)
        if tool["guards"]:
            guard_lines = ["Guards:"]
            guard_lines.extend(f"- {_guard_doc(guard)}" for guard in tool["guards"])
            guard_block = "\n".join(guard_lines)
            base_doc = tool_fn.__doc__ or ""
            tool_fn.__doc__ = f"{base_doc}\n\n{guard_block}" if base_doc else guard_block
        if tool["guards"]:
            if tool_name == "codegen":
                guard_func_name = tool_fn.__name__
            else:
                guard_func_name = f"search_{tool_name}"
            tool_fn = make_guarded_search_tool(
                tool_fn,
                guards=tool["guards"],
                func_name=guard_func_name,
            )
        tools.append(tool_fn)
    return tools
