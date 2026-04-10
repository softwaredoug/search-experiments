from collections import defaultdict, Counter
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from searcharray import SearchArray


def rm3_top_terms(
    arr: SearchArray,
    doc_weights: NDArray[np.float64],
    query_terms: Optional[list[str]],
    top_docs: int,
    originalQueryWeight: float = 0.5,
    num_terms: int = 10,
    binary_relevance: bool = True,
) -> dict[str, float]:
    top_n = np.argsort(-doc_weights)[:top_docs]
    top_n_weights = doc_weights[top_n]

    # Difference from anserini:
    # Anserini takes tf out of norms to also weigh the terms
    # by their importance by doc. Here we weigh basically by
    # proportion of total BM25 score in the doc_weights

    norms = arr.doclengths()[top_n]
    term_to_importance = defaultdict(list)
    all_tfs = 0
    total_termfreqs = np.sum(arr.doclengths())
    for terms, doc_weights, norm in zip(arr[top_n], top_n_weights, norms):
        for term, tf in terms.terms():
            tfs = arr.termfreqs(term)
            background_weight = np.sum(tfs) / total_termfreqs
            if binary_relevance:
                tf = 1
            all_tfs += tf
            foreground_weight = tf / norm
            weight = doc_weights * foreground_weight / (foreground_weight + background_weight)
            term_to_importance[term].append(weight)

    scored_terms = Counter()
    sum_scores = 0
    for term in term_to_importance:
        scored_terms[term] = np.sum(term_to_importance[term])
        sum_scores += scored_terms[term]

    # L1 normalize
    for term in scored_terms:
        if sum_scores > 0:
            scored_terms[term] /= sum_scores

    sum_scorers = 0
    for term in scored_terms:
        p_query_term = 0
        if query_terms and term in query_terms:
            p_query_term = 1 / len(query_terms)
        scored_terms[term] = ((originalQueryWeight * p_query_term) +
                              (1 - originalQueryWeight) * scored_terms[term])
        sum_scorers += scored_terms[term]

    if num_terms is None:
        return scored_terms
    top_terms = scored_terms.most_common(num_terms)
    return {term: importance for term, importance in top_terms}
