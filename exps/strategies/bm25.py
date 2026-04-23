import json
import numpy as np
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer


class BM25Strategy(SearchStrategy):
    _type = "bm25"

    @classmethod
    def build(cls, params: dict, *, corpus, workers: int = 1, **kwargs):
        if "k1" not in params:
            raise ValueError("BM25 config must include 'k1'.")
        if "b" not in params:
            raise ValueError("BM25 config must include 'b'.")
        return cls(corpus, workers=workers, **params)

    def __init__(
        self,
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        k1=1.2,
        b=0.75,
        top_k=10,
        workers=1,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.index = corpus
        self.title_boost = title_boost
        self.description_boost = description_boost
        self.k1 = k1
        self.b = b

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
        similarity = bm25_similarity(k1=self.k1, b=self.b)
        for token in tokenized:
            field_scores = []
            for field, boost in fields.items():
                field_snowball = f"{field}_snowball"
                if field_snowball in self.index:
                    term_match = self.index[field_snowball].array.score(
                        token, similarity=similarity
                    )
                    field_scores.append(term_match * boost)
            term_scores = sum(field_scores) if field_scores else 0
            bm25_scores += term_scores
        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "title_boost": self.title_boost,
            "description_boost": self.description_boost,
            "k1": self.k1,
            "b": self.b,
            "top_k": getattr(self, "top_k", None),
        }
        return json.dumps(payload, sort_keys=True, default=str)
