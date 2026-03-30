import numpy as np
from searcharray import SearchArray

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from .bm25_vect import bm25_rm3_expansion, query_vector


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
        lambd=0.1,
        top_k=10,
        workers=1,
        weigh_query_terms=False
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.index = corpus
        self.title_boost = title_boost
        self.description_boost = description_boost
        self.top_n_terms = top_n_terms
        self.top_n_candidates = top_n_candidates
        self.lambd = lambd
        self.weigh_query_terms = weigh_query_terms

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

    def _rm3_expansion(self, query_terms, doc_weights, field):
        arr = self.index[field].array

        all_terms, exp_vects, exp_top_ns = bm25_rm3_expansion(arr, doc_weights,
                                                              query_terms=query_terms,
                                                              mu=10)
        # Score by summing the frequency of top_ns
        all_top_n_scores = np.zeros(len(self.index))
        traces = {
            5368: [],
            24183: [],
        }
        all_together = zip(all_terms, exp_vects, exp_top_ns)
        # Double counts original currently
        for (term, exp_doc_vect, exp_top_ns) in all_together:
            for doc_weight, top_n in zip(exp_doc_vect, exp_top_ns):
                all_top_n_scores[top_n] += doc_weight
                if top_n in traces:
                    traces[top_n].append((term, doc_weight))
        # Convert to numpy array of len(ccorpus)
        return all_top_n_scores

    def _rm3_field_query(self, field, query, boost):
        arr = self.index[field].array
        query_vect = query_vector(query, snowball_tokenizer, arr)
        # Add b25 score of each term in query_vec, weighed by query_vec weight
        field_scores = np.zeros(len(self.index))
        for term, p_q in query_vect.items():
            scores = arr.score(term)
            if not self.weigh_query_terms:
                p_q = 1.0
            field_scores += (scores * boost * p_q)

        top_n = np.argsort(-field_scores)[:self.top_n_candidates]
        top_n_scores = field_scores[top_n]
        # Thought, this will not take into account whether these are objectively relevant
        as_probabilities = softmax(top_n_scores)
        all_terms, doc_vects, top_ns = bm25_rm3_expansion(arr, top_n, mu=100)
        # Multiply through as_probabilities into all_terms to weigh them
        term_relevances = as_probabilities @ vects  # The relevance model of RM3
        # Add query term relevances into term_relevances
        new_top_ns = []
        for term, p_q in query_vect.items():
            more_docs_scores = arr.score(term)
            new_top_n = np.argsort(-more_docs_scores)[:10]
            new_top_ns.append(new_top_n)
            if term in all_terms:
                idx = all_terms.index(term)
                term_relevances[idx] = (1 - self.lambd) * p_q + (self.lambd * term_relevances[idx])

        # Multiply back into vects matrix to weigh the documents by the relevance model
        doc_scores = term_relevances @ np.log(vects.T)
        doc_scores += np.abs(doc_scores.min()) + 0.0001  # Shift to be positive
        field_scores = 0.0 * field_scores  # Zero out original scores, but keep shape
        field_scores[top_n] = doc_scores

        return field_scores

    def search(self, query, k=10):
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        num_matches = np.zeros(len(self.index))
        for token in tokenized:
            matches = np.zeros(len(self.index), dtype=bool)
            if "title_snowball" in self.index:
                bm25_scores += (
                    self.index["title_snowball"].array.score(token) * self.title_boost
                )
                matches = bm25_scores > 0

            if "description_snowball" in self.index:
                bm25_scores += (
                    self.index["description_snowball"].array.score(token)
                    * self.description_boost
                )
                matches |= bm25_scores > 0
            num_matches += matches.astype(int)
        #all_terms_match = num_matches == len(tokenized)
        doc_weight = bm25_scores.copy()
        #doc_weight[~all_terms_match] = 0

        bm25_scores +=  self._rm3_expansion(tokenized,
                                            doc_weight,
                                            "title_snowball")

        # desc_prf = self._prf_field_query("description_snowball", query, self.description_boost)
        # bm25_scores += desc_prf

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores

        # Doc these two together
