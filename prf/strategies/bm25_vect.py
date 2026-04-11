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
    num_terms: int = 10,
) -> dict[str, float]:
    top_n = np.argsort(-doc_weights)[:top_docs]
    top_n_weights = doc_weights[top_n]

    term_to_importance = defaultdict(list)
    weight_sum = np.sum(top_n_weights)
    for terms, term_importance in zip(arr[top_n], top_n_weights):
        for term, _ in terms.terms():
            weight = term_importance / weight_sum if weight_sum > 0 else 0
            term_to_importance[term].append(weight)

    original_query_weight = 1
    for term in term_to_importance:
        term_to_importance[term] = np.sum(term_to_importance[term])
        if query_terms and term in query_terms:
            term_to_importance[term] += original_query_weight

    # va has more BM25 than champlain?
    # With just the
    if term_to_importance:
        term_to_importance = Counter(term_to_importance)
        term_to_importance = dict(term_to_importance.most_common(num_terms))
    return term_to_importance


def _compute_rm3_vectors(
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
    # 7298534
    for term, term_importance in term_to_importance.items():
        # pwc = arr.docfreq(term) / num_docs
        tfs = arr.termfreqs(term)  # Term freqs in each document

        # Essentialyl the background probability of the term in the corpus
        pwc = np.sum(tfs) / np.sum(doclens)
        if binary_relevance:
            tfs = np.minimum(tfs, 1)
        # The foreground essentially
        # With binary relevance this is dominated by pwc and doclen
        rm3_raw = (tfs + mu * pwc) / (
            doclens + mu
        )  # Term prob in each document with Dirichlet smoothing

        # Essentially makes this a reranker
        rm3_vectors = rm3_raw * doc_weights

        # light and navy blue decorative pillow
        rm3_vectors *= term_importance

        sorted_docs = np.argsort(-rm3_vectors)[:num_docs]

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
    top_terms=10,
    debug_terms: Optional[set[str]] = None,
):
    """Compute sparse vector of the corpus based on terms in the top_n.

    Return a per-document vector representing the probability of the terms in the top_n for that document.
    """
    term_to_importance = _compute_term_importances(
        arr, doc_weights, query_terms, top_docs
    )
    term_to_importance = dict(
        sorted(term_to_importance.items(), key=lambda item: item[1], reverse=True)[:top_terms]
    )

    return _compute_rm3_vectors(
        arr,
        doc_weights,
        term_to_importance,
        binary_relevance,
        mu,
        debug_terms,
    )
