from __future__ import annotations

from typing import Union

import numpy as np
from typing_extensions import Literal

from cheat_at_search.embeddings import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MODEL_NAME,
    load_model,
    load_or_create_embeddings,
)
from cheat_at_search.tokenizers import snowball_tokenizer
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
