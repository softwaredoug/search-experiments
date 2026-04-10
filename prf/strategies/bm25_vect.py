from collections import defaultdict, Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from searcharray import SearchArray


def _compute_term_importances(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    query_terms: Optional[list[str]],
    top_docs: int,
    originalQueryWeight: float = 0.5,
    num_terms: int = 10,
) -> dict[str, float]:
    top_n = np.argsort(-doc_weights)[:top_docs]
    top_n_weights = doc_weights[top_n]

    # Difference from anserini:
    # Anserini takes tf out of norms to also weigh the terms
    # by their importance by doc. Here we weigh basically by
    # proportion of total BM25 score in the doc_weights

    term_to_importance = defaultdict(list)
    weight_sum = np.sum(top_n_weights)
    for terms, term_importance in zip(arr[top_n], top_n_weights):
        for term, _ in terms.terms():
            weight = term_importance / weight_sum if weight_sum > 0 else 0
            term_to_importance[term].append(weight)

    scored_terms = Counter()
    sum_squares = 0
    for term in term_to_importance:
        scored_terms[term] = np.sum(term_to_importance[term])
        p_query_term = 0
        if query_terms and term in query_terms:
            p_query_term = 1 / len(query_terms)
        scored_terms[term] = ((originalQueryWeight * p_query_term) +
                              (1 - originalQueryWeight) * scored_terms[term])
        sum_squares += scored_terms[term] ** 2

    # L2 normalize
    l2_norm = np.sqrt(sum_squares)
    for term in scored_terms:
        if sum_squares > 0:
            scored_terms[term] /= l2_norm

    if num_terms is None:
        return scored_terms
    top_terms = scored_terms.most_common(num_terms)
    return {term: importance for term, importance in top_terms}


def _compute_rm3_vectors(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    term_to_importance: dict[str, float],
    binary_relevance: bool,
    mu: int,
    debug_terms: Optional[set[str]] = None,
):
    doclens = arr.doclengths()
    all_terms = []
    expanded_doc_vects = []
    expanded_top_ns = []
    debug_info = {} if debug_terms else None

    for term, term_importance in term_to_importance.items():
        # pwc = arr.docfreq(term) / num_docs
        tfs = arr.termfreqs(term)  # Term freqs in each document
        pwc = np.sum(tfs) / np.sum(doclens)
        if binary_relevance:
            tfs = np.minimum(tfs, 1)
        # With binary relevance this is dominated by pwc and doclen
        rm3_raw = (tfs + mu * pwc) / (
            doclens + mu
        )  # Term prob in each document with Dirichlet smoothing

        # Original results with this term
        # This essentially defines the foreground
        rm3_vectors = rm3_raw * doc_weights  # Weight by document relevance

        # light and navy blue decorative pillow
        rm3_vectors *= term_importance
        sorted_docs = np.argsort(-rm3_vectors)

        expanded_top_ns.append(sorted_docs)
        expanded_doc_vects.append(rm3_vectors[sorted_docs])

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


def rm3_expansion(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    binary_relevance=True,
    query_terms: Optional[list[str]] = None,
    mu=0,
    top_docs=50,
    debug_terms: Optional[set[str]] = None,
):
    """Compute sparse vector of the corpus based on terms in the top_n.

    Return a per-document vector representing the probability of the terms in the top_n for that document.
    """
    term_to_importance = _compute_term_importances(
        arr, doc_weights, query_terms, top_docs
    )
    return _compute_rm3_vectors(
        arr,
        doc_weights,
        term_to_importance,
        binary_relevance,
        mu,
        debug_terms,
    )
