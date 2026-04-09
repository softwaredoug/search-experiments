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
        weigh_query_terms=False,
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

        if "category_snowball" not in self.index and "category" in corpus:
            self.index["category_snowball"] = SearchArray.index(
                corpus["category"], snowball_tokenizer
            )

    def _rm3_expansion(
        self,
        query_terms,
        doc_weights,
        field,
        binary_relevance=True,
        return_vectors=False,
    ):
        arr = self.index[field].array

        all_terms, exp_vects, exp_top_ns = rm3_expansion(
            arr,
            doc_weights,
            query_terms=query_terms,
            binary_relevance=binary_relevance,
            mu=10,
        )
        # Score by summing the frequency of top_ns
        all_top_n_scores = np.zeros(len(self.index))
        doc_vectors = {} if return_vectors else None
        all_together = zip(all_terms, exp_vects, exp_top_ns)
        for term, exp_doc_vect, exp_top_ns in all_together:
            all_top_n_scores[exp_top_ns] += exp_doc_vect
            if return_vectors:
                for doc_id, score in zip(exp_top_ns, exp_doc_vect):
                    if score == 0:
                        continue
                    doc_vector = doc_vectors.get(doc_id)
                    if doc_vector is None:
                        doc_vector = {}
                        doc_vectors[doc_id] = doc_vector
                    doc_vector[term] = doc_vector.get(term, 0.0) + float(score)
        if return_vectors:
            return all_top_n_scores, doc_vectors
        return all_top_n_scores

    def _search(self, query, k=10, return_vectors=False):
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        num_matches = np.zeros(len(self.index))
        for token in tokenized:
            matches = np.zeros(len(self.index), dtype=bool)
            if "title_snowball" in self.index:
                term_match = self.index["title_snowball"].array.score(token)
                bm25_scores += term_match * self.title_boost
                matches |= term_match > 0

            if "description_snowball" in self.index:
                term_match = self.index["description_snowball"].array.score(token)
                bm25_scores += term_match * self.description_boost
                matches |= term_match > 0
            num_matches += matches.astype(int)
        all_terms_match = num_matches == len(tokenized)
        doc_weight = bm25_scores.copy()
        doc_weight[~all_terms_match] = 0

        if return_vectors:
            title_scores, doc_vectors = self._rm3_expansion(
                tokenized, doc_weight, "title_snowball", return_vectors=True
            )
            bm25_scores += title_scores
            description_scores, description_vectors = self._rm3_expansion(
                tokenized,
                doc_weight,
                "description_snowball",
                binary_relevance=False,
                return_vectors=True,
            )
            bm25_scores += description_scores
            for doc_id, term_scores in description_vectors.items():
                doc_vector = doc_vectors.get(doc_id)
                if doc_vector is None:
                    doc_vectors[doc_id] = term_scores
                    continue
                for term, score in term_scores.items():
                    doc_vector[term] = doc_vector.get(term, 0.0) + score
        else:
            bm25_scores += self._rm3_expansion(
                tokenized, doc_weight, "title_snowball"
            )  # 0.5605
            bm25_scores += self._rm3_expansion(
                tokenized, doc_weight, "description_snowball", binary_relevance=False
            )

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        if return_vectors:
            return doc_vectors, top_k, scores
        return top_k, scores

    def search(self, query, k=10):
        return self._search(query, k=k)

    def vectors(self, query, k=10):
        return self._search(query, k=k, return_vectors=True)

        # Doc these two together
