import numpy as np
from searcharray import SearchArray

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from .bm25_vect import rm3_expansion


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

    def _rm3_expansion(self, query_terms, doc_weights, field,
                       binary_relevance=True):
        arr = self.index[field].array

        all_terms, exp_vects, exp_top_ns = rm3_expansion(arr, doc_weights,
                                                         query_terms=query_terms,
                                                         binary_relevance=binary_relevance,
                                                         mu=10)
        # Score by summing the frequency of top_ns
        all_top_n_scores = np.zeros(len(self.index))
        all_together = zip(all_terms, exp_vects, exp_top_ns)
        for (term, exp_doc_vect, exp_top_ns) in all_together:
            all_top_n_scores[exp_top_ns] += exp_doc_vect
        return all_top_n_scores

    def search(self, query, k=10):
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        num_matches = np.zeros(len(self.index))
        for token in tokenized:
            matches = np.zeros(len(self.index), dtype=bool)
            if "title_snowball" in self.index:
                term_match = self.index["title_snowball"].array.score(token)
                bm25_scores += (
                    term_match * self.title_boost
                )
                matches |= term_match > 0

            if "description_snowball" in self.index:
                term_match = self.index["description_snowball"].array.score(token)
                bm25_scores += (
                    term_match
                    * self.description_boost
                )
                matches |= term_match > 0
            num_matches += matches.astype(int)
        all_terms_match = num_matches == len(tokenized)
        doc_weight = bm25_scores.copy()
        doc_weight[~all_terms_match] = 0

        bm25_scores += self._rm3_expansion(tokenized,
                                           doc_weight,
                                           "title_snowball")  # 0.5605
        bm25_scores += self._rm3_expansion(tokenized,
                                           doc_weight,
                                           "description_snowball",
                                           binary_relevance=False)

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores

        # Doc these two together
