import numpy as np
from searcharray import SearchArray
from collections import Counter

from cheat_at_search.strategy import SearchStrategy
from searcharray.similarity import compute_idf
from cheat_at_search.tokenizers import snowball_tokenizer
from .prf_rerank_terms import (
    compute_bm25_matrix,
    term_space_eigenvectors,
    weighed_bm25_search,
)

# Good / bad eigenvectors for queries we know about, just to see how they help
labeled_eigenvectsors = {
    "what county is champlain va in?": [0, 1, 2, 3, 4],
}


class PRFExpand(SearchStrategy):
    def __init__(
        self,
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        bm25_k1=1.2,
        bm25_b=0.75,
        top_n_terms=10,
        top_n_eigenvectors=10,
        top_n_candidates=100,
        top_terms_per_eigenvector=20,
        top_k=10,
        workers=1,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.index = corpus
        self.title_boost = title_boost
        self.description_boost = description_boost
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        self.top_n_terms = top_n_terms
        self.top_n_eigenvectors = top_n_eigenvectors
        self.top_n_candidates = top_n_candidates
        self.top_terms_per_eigenvector = top_terms_per_eigenvector

        for field in ["title", "description"]:
            snowball_field = f"{field}_snowball"
            if snowball_field not in corpus:
                raise ValueError(f"RM3 field '{field}' not found in corpus")
                self.index[snowball_field] = SearchArray.index(
                    corpus[field], snowball_tokenizer
                )

    def _search(self, query, k=10, return_vectors=False, debug_terms=None):
        tokenized = snowball_tokenizer(query)
        bm25_scores, doc_weight = weighed_bm25_search(
            corpus=self.corpus,
            fields={"title": self.title_boost, "description": self.description_boost},
            query_terms=tokenized,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
        )

        matrix = compute_bm25_matrix(
            self.corpus["description_snowball"].array,
            doc_weights=bm25_scores,
            top_docs=self.top_n_candidates,
        )
        eigenvalues, eigenvectors = term_space_eigenvectors(
            matrix,
            top_r=self.top_n_eigenvectors,
        )
        pca_eigenvalues, pca_eigenvectors = term_space_eigenvectors(
            matrix,
            top_r=self.top_n_eigenvectors,
        )
        max_bm25 = np.max(bm25_scores)
        for vec_idx, (eigenvalue, eigenvector) in enumerate(zip(pca_eigenvalues, pca_eigenvectors)):
            top_terms = sorted(
                eigenvector.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[: self.top_terms_per_eigenvector]
            top_terms = [(term, abs(weight)) for term, weight in top_terms]
            for term, weight in top_terms:
                print(f"PCA {vec_idx} -- {term}: {weight} (eigenvalue: {eigenvalue})")

                term_scores = self.corpus["description_snowball"].array.score(term)
                factors = (term_scores > 0) * weight * (1 + eigenvalue)
                if vec_idx in [0]:
                    bm25_scores += factors
            print("--")
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores

        # exp_scores = np.zeros_like(bm25_scores)
        for vec_idx, (eigenvalue, eigenvector) in enumerate(zip(eigenvalues, eigenvectors)):
            if not eigenvector:
                continue
            top_terms = sorted(
                eigenvector.items(),
                key=lambda item: abs(item[1]),
                reverse=True,
            )[: self.top_terms_per_eigenvector]
            top_terms = [(term, abs(weight)) for term, weight in top_terms]
            print(f" {vec_idx} --")
            for term, weight in top_terms:
                term_scores = self.corpus["description_snowball"].array.score(term)
                factors = (term_scores > 0) * weight * (1 + eigenvalue) * (100 + max_bm25)
                if vec_idx in [1, 5, 8, 9]:
                    bm25_scores -= factors
                else:
                    bm25_scores += factors
                print(term, weight, eigenvalue, factors.max())
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores

    def search(self, query, k=10):
        return self._search(query, k=k)
