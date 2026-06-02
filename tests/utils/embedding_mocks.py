from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cheat_at_search.tokenizers import snowball_tokenizer
from searcharray import SearchArray


@dataclass
class MockEmbeddingModel:
    query_vectors: dict[str, np.ndarray]
    dim: int = 3

    def encode(self, inputs: str | Iterable[str]):
        if isinstance(inputs, str):
            return self._encode_single(inputs)
        vectors = [self._encode_single(text) for text in inputs]
        return np.vstack(vectors)

    def _encode_single(self, text: str) -> np.ndarray:
        if text in self.query_vectors:
            return self.query_vectors[text]
        terms = snowball_tokenizer(text)
        if not terms:
            return np.zeros(self.dim)
        vectors = [self.query_vectors.get(term) for term in terms if term in self.query_vectors]
        if not vectors:
            return np.zeros(self.dim)
        return np.mean(np.vstack(vectors), axis=0)


def build_query_vectors(
    queries: Iterable[str],
    *,
    dim: int = 3,
    seed: int = 123,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    term_vectors: dict[str, np.ndarray] = {}
    query_vectors: dict[str, np.ndarray] = {}
    for query in queries:
        terms = snowball_tokenizer(query)
        if not terms:
            query_vectors[query] = np.zeros(dim)
            continue
        vectors = []
        for term in terms:
            if term not in term_vectors:
                term_vectors[term] = rng.random(dim)
            vectors.append(term_vectors[term])
        query_vectors[query] = np.mean(np.vstack(vectors), axis=0)
    return query_vectors, term_vectors


def build_mock_embeddings(
    corpus,
    judgments,
    passage_fn,
    *,
    dim: int = 3,
    seed: int = 123,
    doc_base_weight: float = 0.1,
    term_weight: float = 1.0,
) -> tuple[np.ndarray, MockEmbeddingModel]:
    queries = judgments["query"].dropna().astype(str).unique().tolist()
    query_vectors, term_vectors = build_query_vectors(queries, dim=dim, seed=seed)
    rng = np.random.default_rng(seed)

    if "title_snowball" not in corpus.columns and "title" in corpus.columns:
        corpus["title_snowball"] = SearchArray.index(corpus["title"], snowball_tokenizer)
    if "description_snowball" not in corpus.columns and "description" in corpus.columns:
        corpus["description_snowball"] = SearchArray.index(corpus["description"], snowball_tokenizer)

    num_docs = len(corpus)
    doc_vectors = rng.random((num_docs, dim)) * doc_base_weight

    title_scores = None
    if "title_snowball" in corpus.columns:
        title_scores = corpus["title_snowball"].array
    description_scores = None
    if "description_snowball" in corpus.columns:
        description_scores = corpus["description_snowball"].array

    for term, term_vector in term_vectors.items():
        scores = None
        if title_scores is not None:
            scores = title_scores.score(term)
        if description_scores is not None:
            desc_score = description_scores.score(term)
            scores = desc_score if scores is None else scores + desc_score
        if scores is None:
            continue
        mask = scores > 0
        if np.any(mask):
            doc_vectors[mask] += term_weight * term_vector

    embeddings = doc_vectors if num_docs else np.zeros((0, dim))
    return embeddings, MockEmbeddingModel(query_vectors=query_vectors, dim=dim)


def mock_load_or_create_embeddings(
    corpus,
    judgments,
    passage_fn,
    *,
    dim: int = 3,
    seed: int = 123,
    doc_base_weight: float = 0.1,
    term_weight: float = 1.0,
    **_kwargs,
):
    return build_mock_embeddings(
        corpus,
        judgments,
        passage_fn,
        dim=dim,
        seed=seed,
        doc_base_weight=doc_base_weight,
        term_weight=term_weight,
    )
