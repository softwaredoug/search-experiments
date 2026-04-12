import numpy as np
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity, compute_idf

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer


class DoubleIDFBM25Strategy(SearchStrategy):
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
        bm25_scores = np.zeros(len(self.index))
        similarity = bm25_similarity(k1=self.bm25_k1, b=self.bm25_b)
        for token in tokenized:
            field_scores = []
            field_dfs = []
            for field, boost in fields.items():
                field_snowball = f"{field}_snowball"
                if field_snowball in self.index:
                    term_match = self.index[field_snowball].array.score(
                        token, similarity=similarity
                    )
                    field_dfs.append(self.index[field_snowball].array.docfreq(token))
                    field_scores.append(term_match * boost)
            df = max(field_dfs) if field_dfs else 0
            term_scores = sum(field_scores) if field_scores else 0
            term_scores *= compute_idf(len(self.index), df)
            bm25_scores += term_scores
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
