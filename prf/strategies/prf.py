from collections import Counter

import numpy as np
from searcharray import SearchArray

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer


class PRFStrategy(SearchStrategy):
    def __init__(
        self,
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        top_k=10,
        workers=1,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.index = corpus
        self.title_boost = title_boost
        self.description_boost = description_boost

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
        bm25_scores = np.zeros(len(self.index))
        if "title_snowball" in self.index:
            for token in tokenized:
                bm25_scores += (
                    self.index["title_snowball"].array.score(token) * self.title_boost
                )
        if "description_snowball" in self.index:
            for token in tokenized:
                bm25_scores += (
                    self.index["description_snowball"].array.score(token)
                    * self.description_boost
                )

        top_k_candidates = np.argsort(-bm25_scores)[:50]
        non_zero = bm25_scores[top_k_candidates] > 0
        filtered_candidates = top_k_candidates[non_zero]
        titles = []
        if "title" in self.index and len(filtered_candidates) > 0:
            titles = (
                self.index.iloc[filtered_candidates]["title"]
                .fillna("")
                .astype(str)
                .tolist()
            )

        title_text = " ".join(titles)
        prf_tokens = snowball_tokenizer(title_text)
        token_counts = Counter(prf_tokens)
        top_tokens = token_counts.most_common(10)

        print(f"PRF tokens for query: {query}")
        if top_tokens:
            for token, count in top_tokens:
                print(f"{token}: {count}")
        else:
            print("No PRF tokens found.")

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        return top_k, scores
