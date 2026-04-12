import numpy as np
from searcharray import SearchArray

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from prf.strategies.prf_rerank_terms import weighed_bm25_search


class BM25Strategy(SearchStrategy):
    def __init__(
        self,
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        bm25_k1=1.2,
        bm25_b=0.75,
        top_k=10,
        workers=1,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.index = corpus
        self.title_boost = title_boost
        self.description_boost = description_boost
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        if "title_snowball" not in self.index and "title" in corpus:
            self.index["title_snowball"] = SearchArray.index(
                corpus["title"], snowball_tokenizer
            )
        if "description_snowball" not in self.index and "description" in corpus:
            self.index["description_snowball"] = SearchArray.index(
                corpus["description"], snowball_tokenizer
            )

    def search(self, query, k=10):
        tokenized = snowball_tokenizer(query)
        fields = {
            "title": self.title_boost,
            "description": self.description_boost,
        }
        bm25_scores, _ = weighed_bm25_search(
            self.index,
            fields,
            tokenized,
            bm25_k1=self.bm25_k1,
            bm25_b=self.bm25_b,
        )
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
