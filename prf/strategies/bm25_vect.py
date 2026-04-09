from collections import defaultdict
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from searcharray import SearchArray


def rm3_expansion(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    binary_relevance=True,
    query_terms: Optional[list[str]] = None,
    mu=0,
    top_docs=50
):
    """Compute sparse vector of the corpus based on terms in the top_n.

    Return a per-document vector representing the probability of the terms in the top_n for that document.
    """
    # Score term in every
    top_n = np.argsort(-doc_weights)[:top_docs]
    top_n_weights = doc_weights[top_n]
    doclens = arr.doclengths()

    all_terms = []
    expanded_doc_vects = []
    expanded_top_ns = []
    term_pwcs = {}
    term_to_importance = defaultdict(list)
    timp_log = defaultdict(list)
    # First we iterate the forward index of the top_n
    # to get the importance of each term in the top_n
    # The sum of all the doc weights (BM25 scores) it occurs in
    weight_sum = np.sum(top_n_weights)
    for doc_id, terms, term_importance in zip(top_n, arr[top_n], top_n_weights):
        for term, _ in terms.terms():
            weight = term_importance / weight_sum if weight_sum > 0 else 0
            term_to_importance[term].append(weight)
            timp_log[term].append((weight, doc_id))

    # Collapse to mean
    original_query_weight = 1
    for term in term_to_importance:
        term_to_importance[term] = np.sum(term_to_importance[term])
        if term in query_terms:
            term_to_importance[term] += original_query_weight

    for term, term_importance in term_to_importance.items():
        # pwc = arr.docfreq(term) / num_docs
        tfs = arr.termfreqs(term)  # Term freqs in each document
        pwc = np.sum(tfs) / np.sum(doclens)
        if binary_relevance:
            tfs = np.minimum(tfs, 1)
        # With binary relevance this is dominated by pwc and doclen
        rm3_vectors = (tfs + mu * pwc) / (doclens + mu)  # Term prob in each document with Dirichlet smoothing
        term_pwcs[term] = pwc

        sorted_docs = np.argsort(-rm3_vectors)
        # Original results with this term
        # This essentially defines the foreground
        rm3_vectors *= doc_weights  # Weight by document relevance

        # light and navy blue decorative pillow
        rm3_vectors *= term_importance
        sorted_docs = np.argsort(-rm3_vectors)

        expanded_top_ns.append(sorted_docs)
        expanded_doc_vects.append(rm3_vectors[sorted_docs])

        # Add to results
        all_terms.append(term)

    return all_terms, expanded_doc_vects, expanded_top_ns
