import numpy as np
from searcharray import SearchArray

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from .bm25_vect import bm25_prf_vect, query_vector


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PRFStrategy(SearchStrategy):
    def __init__(
        self,
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        top_n_terms=10,
        top_n_candidates=50,
        lambd=0.5,
        top_k=10,
        workers=1,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.index = corpus
        self.title_boost = title_boost
        self.description_boost = description_boost
        self.top_n_terms = top_n_terms
        self.top_n_candidates = top_n_candidates
        self.lambd = lambd

        if "title_snowball" not in self.index and "title" in corpus:
            self.index["title_snowball"] = SearchArray.index(
                corpus["title"], snowball_tokenizer
            )
        if "description_snowball" not in self.index and "description" in corpus:
            self.index["description_snowball"] = SearchArray.index(
                corpus["description"], snowball_tokenizer
            )

        if 'category_snowball' not in self.index and 'category' in corpus:
            self.index['category_snowball'] = SearchArray.index(
                corpus['category'], snowball_tokenizer
            )

    def _prf_field_query(self, field, query, boost):
        arr = self.index[field].array
        query_vect = query_vector(query, snowball_tokenizer, arr)
        # Add b25 score of each term in query_vec, weighed by query_vec weight
        field_scores = np.zeros(len(self.index))
        for term, p_q in query_vect.items():
            scores = arr.score(term)
            field_scores += (scores * boost)

        top_n = np.argsort(-field_scores)[:self.top_n_candidates]
        top_n_scores = field_scores[top_n]
        # Thought, this will not take into account whether these are objectively relevant
        as_probabilities = softmax(top_n_scores)
        all_terms, vects = bm25_prf_vect(arr, top_n, mu=1000)
        # Multiply through as_probabilities into all_terms to weigh them
        term_relevances = as_probabilities @ vects  # The relevance model of RM3
        # Add query term relevances into term_relevances
        for term, p_q in query_vect.items():
            if term in all_terms:
                idx = all_terms.index(term)
                term_relevances[idx] = (1 - self.lambd) * p_q + (self.lambd * term_relevances[idx])

        # Multiply back into vects matrix to weigh the documents by the relevance model
        doc_scores = np.log(term_relevances) @ vects.T
        doc_scores += doc_scores.min() + 0.0001  # Shift to be positive
        field_scores = 0.0 * field_scores  # Zero out original scores, but keep shape
        field_scores[top_n] = doc_scores

        return field_scores

    def search(self, query, k=10):
        bm25_scores = self._prf_field_query("title_snowball", query, self.title_boost)

        desc_prf = self._prf_field_query("description_snowball", query, self.description_boost)
        bm25_scores += desc_prf

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores

        # Doc these two together
