from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from searcharray import SearchArray


def l2_norm(vector: Counter) -> float:
    """Calculate the L2 norm of a sparse vector represented as a Counter."""
    return np.sqrt(sum(value ** 2 for value in vector.values()))


def mean_vector(vectors: list[Counter], weights: list[float]) -> Counter:
    """Calculate the mean vector from a list of sparse vectors represented as Counters."""
    mean_vec = Counter()
    for weight, vec in zip(weights, vectors):
        for term, value in vec.items():
            mean_vec[term] += weight * value
    # Divide by the number of vectors to get the mean
    num_vectors = len(vectors)
    if num_vectors > 0:
        for term in mean_vec:
            mean_vec[term] /= num_vectors
    return mean_vec


def dot_product(vec1: Counter, vec2: Counter) -> float:
    """Calculate the dot product of two sparse vectors represented as Counters."""
    return sum(vec1[term] * vec2[term] for term in vec1 if term in vec2)


def to_dict(vector: np.ndarray, terms: list[str]) -> Counter:
    """Convert a dense vector and corresponding terms into a sparse vector represented as a Counter."""
    return Counter({term: vector[i] for i, term in enumerate(terms) if vector[i] != 0})


def query_vector(query: str, tokenizer, arr: SearchArray) -> Counter:
    vector = Counter()
    num_docs = len(arr)
    for term in tokenizer(query):
        df = arr.docfreq(term)
        idf = np.log((num_docs - df + 0.5) / (df + 0.5) + 1)
        vector[term] += idf

    summed_idf = sum(vector.values())
    if summed_idf > 0:
        for term in vector:
            vector[term] /= summed_idf
    return vector


# low profile loveseat recliner
from cheat_at_search.wands_data import corpus


def bm25_rm3_expansion(
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
    num_docs = len(arr)
    all_terms = []
    expanded_doc_vects = []
    expanded_top_ns = []
    term_pwcs = {}
    term_to_importance = defaultdict(list)
    timp_log = defaultdict(list)
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
        # Term prob in collection (an approximate prior compatible with binary relevance)
        df = arr.docfreq(term)
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

        # 203


        # light and navy blue decorative pillow
        rm3_vectors *= term_importance
        sorted_docs = np.argsort(-rm3_vectors)

        expanded_top_ns.append(sorted_docs)
        expanded_doc_vects.append(rm3_vectors[sorted_docs])

        # Add to results
        all_terms.append(term)

    term_importances = sorted(list(term_to_importance.items()), key=lambda x: -x[1])
    term_weights = np.sum(np.stack(expanded_doc_vects), axis=1)
    num_docs_with_term = np.sum([np.sum(vect > 0) for vect in expanded_doc_vects])
    term_weights /= num_docs_with_term if num_docs_with_term > 0 else 1
    weights = sorted(list(zip(all_terms, term_weights)), key=lambda x: -x[1])
    pwcs = sorted(term_pwcs.items(), key=lambda x: -x[1])
    # low profile loveseat recliner

    return all_terms, expanded_doc_vects, expanded_top_ns


def bm25_doc_vects(
    arr: SearchArray,
    prf_vector: Counter,
    terms: list[str]
) -> NDArray[np.float64]:

    term_scores_per_doc = []
    for term in terms:
        doc_scores = arr.score(term)
        term_scores_per_doc.append(doc_scores)
    # Reshape into matrix
    term_scores_matrix = np.stack(term_scores_per_doc, axis=1)
    # Normalize by l1 norm rowwise
    norms = np.linalg.norm(term_scores_matrix, axis=1, keepdims=True)
    doc_indices_match = np.where(norms)[0]
    # 2851 -> 9529
    matching_term_scores_matrix = term_scores_matrix[doc_indices_match]
    matching_norms = norms[doc_indices_match]
    term_scores_matrix_normed = matching_term_scores_matrix / matching_norms
    return doc_indices_match, term_scores_matrix_normed
