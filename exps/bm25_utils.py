from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from searcharray.similarity import bm25_similarity, compute_idf


def _bm25_search_stats(
    corpus,
    fields: dict[str, float],
    query_terms: list[str],
    *,
    double_idf: bool = False,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, int]]:
    bm25_scores = np.zeros(len(corpus))
    num_matches = np.zeros(len(corpus))
    df_weights = np.zeros(len(corpus))
    term_dfs: dict[str, int] = {}
    similarity = bm25_similarity(k1=bm25_k1, b=bm25_b)
    for token in query_terms:
        matches = np.zeros(len(corpus), dtype=bool)
        field_dfs = []
        field_scores = []
        for field, boost in fields.items():
            field_snowball = f"{field}_snowball"
            if field_snowball in corpus:
                term_match = corpus[field_snowball].array.score(
                    token, similarity=similarity
                )
                matches |= term_match > 0
                field_df = corpus[field_snowball].array.docfreq(token)
                field_dfs.append(field_df)
                field_scores.append(term_match * boost)
        df = max(field_dfs) if field_dfs else 0
        term_scores = sum(field_scores) if field_scores else 0
        if double_idf:
            term_scores *= compute_idf(len(corpus), df)
        bm25_scores += term_scores

        term_dfs[token] = df
        df_weights[matches] += compute_idf(len(corpus), df)
        num_matches += matches.astype(int)
    doc_weight = bm25_scores.copy()
    doc_weight *= df_weights
    return bm25_scores, doc_weight, num_matches, term_dfs


def bm25_search_details(
    corpus,
    fields: dict[str, float],
    query_terms: list[str],
    *,
    double_idf: bool = False,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, int]]:
    return _bm25_search_stats(
        corpus,
        fields,
        query_terms,
        double_idf=double_idf,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
