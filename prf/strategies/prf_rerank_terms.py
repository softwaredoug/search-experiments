from collections import defaultdict, Counter
from typing import Optional, Tuple
from searcharray.similarity import bm25_similarity, compute_idf

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from searcharray import SearchArray


def _bm25_search_stats(
    corpus: pd.DataFrame,
    fields: dict[str, float],
    query_terms: list[str],
    double_idf: bool = False,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> Tuple[NDArray[float], NDArray[float], NDArray[float], dict[str, int]]:
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


def weighed_bm25_search(
    corpus: pd.DataFrame,
    fields: dict[str, float],
    query_terms: list[str],
    double_idf: bool = False,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> Tuple[NDArray[float], NDArray[float]]:
    """BM25 scores over a list of fields, with weighting based on IDF and number of matches."""
    bm25_scores, doc_weight, _, _ = _bm25_search_stats(
        corpus,
        fields,
        query_terms,
        double_idf=double_idf,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    return bm25_scores, doc_weight


def bm25_search_details(
    corpus: pd.DataFrame,
    fields: dict[str, float],
    query_terms: list[str],
    double_idf: bool = False,
    bm25_k1: float = 1.2,
    bm25_b: float = 0.75,
) -> Tuple[NDArray[float], NDArray[float], NDArray[float], dict[str, int]]:
    return _bm25_search_stats(
        corpus,
        fields,
        query_terms,
        double_idf=double_idf,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )


def _compute_bm25(tf, doclen, df, num_docs, k1=1.2, b=0.75):
    idf = compute_idf(num_docs, df)
    avg_doclen = doclen.mean()
    norm_tf = tf / (tf + k1 * (1 - b + b * (doclen / avg_doclen)))
    return norm_tf * idf


def compute_bm25_matrix(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    top_docs: int,
) -> dict[str, list[tuple[int, float]]]:
    top_n = np.argsort(-doc_weights)[:top_docs]
    top_n_weights = doc_weights[top_n]

    term_to_importance = defaultdict(list)
    weight_sum = np.sum(top_n_weights)
    for doc_id, terms, term_importance in zip(top_n, arr[top_n], top_n_weights):
        this_doc = Counter()
        for term, tf in terms.terms():
            term_df = arr.docfreq(term)
            weight = _compute_bm25(
                tf=tf,
                doclen=arr.doclengths().mean(),
                df=term_df,
                num_docs=len(arr),
                k1=1.2,
                b=0.75,
            )
            orig_bm25_weight = term_importance / weight_sum if weight_sum > 0 else 0
            weight = (orig_bm25_weight**2) * weight
            this_doc[term] += weight
            term_to_importance[term].append((doc_id, weight))
    return term_to_importance


def term_space_eigenvectors(
    term_to_importance: dict[str, list[tuple[int, float]]],
    top_r: int,
    *,
    min_abs_weight: float = 0.0,
) -> tuple[list[float], list[dict[str, float]]]:
    if top_r <= 0:
        raise ValueError("top_r must be positive")
    if not term_to_importance:
        return [], []

    terms = sorted(term_to_importance)
    doc_ids = sorted(
        {doc_id for entries in term_to_importance.values() for doc_id, _ in entries}
    )
    if not doc_ids:
        return [], []

    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    matrix = np.zeros((len(terms), len(doc_ids)), dtype=float)
    for term_idx, term in enumerate(terms):
        for doc_id, weight in term_to_importance[term]:
            matrix[term_idx, doc_index[doc_id]] += weight

    gram = matrix @ matrix.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(-eigenvalues)
    top_indices = order[: min(top_r, len(terms))]

    eigenvalues_out = []
    sparse_vectors = []
    for idx in top_indices:
        vector = eigenvectors[:, idx]
        sparse_vector = {
            term: float(weight)
            for term, weight in zip(terms, vector)
            if abs(weight) > min_abs_weight
        }
        eigenvalues_out.append(float(eigenvalues[idx]))
        sparse_vectors.append(sparse_vector)
    return eigenvalues_out, sparse_vectors


def term_space_pca(
    term_to_importance: dict[str, list[tuple[int, float]]],
    top_r: int,
    *,
    min_abs_weight: float = 0.0,
) -> tuple[list[float], list[dict[str, float]]]:
    if top_r <= 0:
        raise ValueError("top_r must be positive")
    if not term_to_importance:
        return [], []

    terms = sorted(term_to_importance)
    doc_ids = sorted(
        {doc_id for entries in term_to_importance.values() for doc_id, _ in entries}
    )
    if not doc_ids:
        return [], []

    doc_index = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    matrix = np.zeros((len(terms), len(doc_ids)), dtype=float)
    for term_idx, term in enumerate(terms):
        for doc_id, weight in term_to_importance[term]:
            matrix[term_idx, doc_index[doc_id]] += weight

    centered = matrix - matrix.mean(axis=1, keepdims=True)
    denom = len(doc_ids) - 1
    if denom <= 0:
        denom = 1
    cov = (centered @ centered.T) / denom
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(-eigenvalues)
    top_indices = order[: min(top_r, len(terms))]

    eigenvalues_out = []
    sparse_vectors = []
    for idx in top_indices:
        vector = eigenvectors[:, idx]
        sparse_vector = {
            term: float(weight)
            for term, weight in zip(terms, vector)
            if abs(weight) > min_abs_weight
        }
        eigenvalues_out.append(float(eigenvalues[idx]))
        sparse_vectors.append(sparse_vector)
    return eigenvalues_out, sparse_vectors


def topn_sparse(arr, doc_weights, query_terms=None, top_docs=50, top_terms=40):
    sparse_terms = compute_bm25_matrix(arr, doc_weights, top_docs)
    if not sparse_terms:
        return {}

    term_to_importance = {}
    sum_bm25s = 0.0
    for weighed_docs in sparse_terms.values():
        for _, weight in weighed_docs:
            sum_bm25s += weight
    for term, weighed_docs in sparse_terms.items():
        weights = [weighed_doc[1] for weighed_doc in weighed_docs]
        term_weight = np.sum(weights)
        if sum_bm25s > 0:
            term_weight /= sum_bm25s
        term_to_importance[term] = term_weight

    if term_to_importance:
        required_in_result = {}
        for term in query_terms or []:
            if term in term_to_importance:
                required_in_result[term] = term_to_importance[term]
        term_to_importance = Counter(term_to_importance)
        term_to_importance = dict(term_to_importance.most_common(top_terms))
        term_to_importance.update(required_in_result)
    return l1_normalize(term_to_importance)


def _rel_term_strengths(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    term_to_importance: dict[str, float],
    binary_relevance: bool,
    mu: int,
    debug_terms: Optional[set[str]] = None,
    num_docs: Optional[int] = None,
):
    doclens = arr.doclengths()
    all_terms = []
    expanded_doc_vects = []
    expanded_top_ns = []
    debug_info = {} if debug_terms else None
    max_weight = doc_weights.max()
    doc_weights[doc_weights < max_weight * 0.1] = 0
    live_docs = np.where(doc_weights > 0)[0]
    # 7298534
    for term, term_importance in term_to_importance.items():
        # pwc = arr.docfreq(term) / num_docs
        tfs = arr.termfreqs(term)  # Term freqs in each document

        # Essentialyl the background probability of the term in the corpus
        pwc = np.sum(tfs) / np.sum(doclens)
        if binary_relevance:
            tfs = np.minimum(tfs, 1)

        # tf strength relative to background
        rm3_raw = (tfs + mu * pwc) / (
            doclens + mu
        )  # Term prob in each document with Dirichlet smoothing

        # Essentially makes this a reranker within BM25 matches
        rm3_vectors = rm3_raw * doc_weights

        # 7287868
        rm3_vectors *= term_importance

        vect = rm3_vectors[live_docs]
        # sorted_subvect_indices = np.argsort(-vect)
        # sorted_docs = live_docs[sorted_subvect_indices][:num_docs]

        expanded_top_ns.append(live_docs)
        expanded_doc_vects.append(vect)

        # Add to results
        all_terms.append(term)

        if debug_terms is not None and term in debug_terms:
            debug_info[term] = {
                "term": term,
                "term_importance": float(term_importance),
                "pwc": float(pwc),
                "rm3_raw": rm3_raw,
                "rm3_after_importance": rm3_raw * term_importance,
                "rm3_after_doc_weights": rm3_raw * term_importance * doc_weights,
            }
    return all_terms, expanded_doc_vects, expanded_top_ns, debug_info


def l1_normalize(sparse_vector: dict[str, float]) -> dict[str, float]:
    total = sum(sparse_vector.values())
    if total > 0:
        return {term: weight / total for term, weight in sparse_vector.items()}
    return sparse_vector


def top_n_term_strengths(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    binary_relevance=True,
    query_terms: Optional[list[str]] = None,
    mu=0,
    top_docs=50,
    top_terms=40,
    debug_terms: Optional[set[str]] = None,
):
    """Compute sparse vector of the corpus based on terms in the top_n.

    Return a per-document vector representing the probability of the terms in the top_n for that document.
    """
    term_to_importance = topn_sparse(
        arr,
        doc_weights,
        query_terms=query_terms,
        top_docs=top_docs,
        top_terms=top_terms,
    )

    return _rel_term_strengths(
        arr,
        doc_weights,
        term_to_importance,
        binary_relevance,
        mu,
        debug_terms,
        num_docs=top_docs,
    )
