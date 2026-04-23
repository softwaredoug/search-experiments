import numpy as np
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity, compute_idf

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer


# Full queries
# MSMarco (n=6980)
# mean_mrr_bm25=0.1805
# mean_mrr_bm25_reweighed=0.1879

# ESCI (n=1000)
# mean_ndcg_bm25=0.2895
# median_ndcg_bm25=0.1707
# mean_ndcg_bm25_reweighed=0.3265

# WANDS (n=480)
# mean_ndcg_bm25=0.5408
# median_ndcg_bm25=0.4746
# mean_ndcg_bm25_reweighed=0.5613


class ReweighedBM25Strategy(SearchStrategy):
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
        df_weights = np.zeros(len(self.index))
        similarity = bm25_similarity(k1=self.k1, b=self.b)
        for token in tokenized:
            matches = np.zeros(len(self.index), dtype=bool)
            field_dfs = []
            field_scores = []
            for field, boost in fields.items():
                field_snowball = f"{field}_snowball"
                if field_snowball in self.index:
                    term_match = self.index[field_snowball].array.score(
                        token, similarity=similarity
                    )
                    matches |= term_match > 0
                    field_dfs.append(self.index[field_snowball].array.docfreq(token))
                    field_scores.append(term_match * boost)
            df = max(field_dfs) if field_dfs else 0
            term_scores = sum(field_scores) if field_scores else 0
            bm25_scores += term_scores
            df_weights[matches] += compute_idf(len(self.index), df)
        doc_weight = bm25_scores * df_weights
        top_k = np.argsort(-doc_weight)[:k]
        scores = doc_weight[top_k]
        return top_k, scores
