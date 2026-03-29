from collections import Counter
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


def bm25_prf_vect(
    arr: SearchArray,
    top_n: NDArray[np.int_],
    mu=0
):
    """Get BM25 vector for all matches=True."""
    # Score term in every
    doclens = arr.doclengths()
    num_docs = len(arr)
    all_terms = []
    vectors = []
    for terms in arr[top_n]:
        for term, _ in terms.terms():
            if term in all_terms:
                continue
            pwc = arr.docfreq(term) / num_docs
            bm25_vectors = (arr.termfreqs(term) + mu * pwc) / (doclens + mu)
            vectors.append(bm25_vectors[top_n])
            all_terms.append(term)
    return all_terms, np.stack(vectors, axis=1)


from cheat_at_search.wands_data import corpus


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
