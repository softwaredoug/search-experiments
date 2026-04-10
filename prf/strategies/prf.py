import numpy as np
from searcharray import SearchArray

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from .bm25_vect import rm3_top_terms


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class PRFStrategy(SearchStrategy):
    def __init__(
        self,
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        rm3_fields=None,
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
        self.rm3_fields = self._normalize_rm3_fields(rm3_fields)
        self.top_n_terms = top_n_terms
        self.top_n_candidates = top_n_candidates
        self.lambd = lambd
        self.weigh_query_terms = weigh_query_terms

        for field in self.rm3_fields:
            snowball_field = f"{field}_snowball"
            if snowball_field not in self.index:
                if field not in corpus:
                    raise ValueError(f"RM3 field '{field}' not found in corpus")
                self.index[snowball_field] = SearchArray.index(
                    corpus[field], snowball_tokenizer
                )

    def _normalize_rm3_fields(self, rm3_fields):
        if rm3_fields is None:
            return ["title", "description"]
        allowed_fields = {"title", "description", "category"}
        if isinstance(rm3_fields, str):
            rm3_fields = [
                field.strip() for field in rm3_fields.split(",") if field.strip()
            ]
        if not rm3_fields:
            raise ValueError("RM3 fields cannot be empty")
        unknown_fields = [field for field in rm3_fields if field not in allowed_fields]
        if unknown_fields:
            unknown_fields_str = ", ".join(sorted(unknown_fields))
            allowed_fields_str = ", ".join(sorted(allowed_fields))
            raise ValueError(
                f"Unknown RM3 fields: {unknown_fields_str}. Allowed: {allowed_fields_str}"
            )
        return list(dict.fromkeys(rm3_fields))

    def expansion_terms(
        self,
        query_terms,
        doc_weights,
        field,
        binary_relevance=True,
        return_vectors=False,
    ):
        arr = self.index[field].array

        return rm3_top_terms(
            arr,
            doc_weights,
            query_terms if self.weigh_query_terms else None,
            top_docs=self.top_n_candidates,
            originalQueryWeight=1 - self.lambd,
            num_terms=self.top_n_terms,
            binary_relevance=binary_relevance,
        )

    def search(self, query, k=10):
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

        rm3_scores = np.zeros(len(self.index))

        for field in ['description']:
            weighed_terms = self.expansion_terms(
                tokenized,
                doc_weight,
                f"{field}_snowball",
                binary_relevance=field != "description",
            )
            for term, weight in weighed_terms.items():
                if f"{field}_snowball" in self.index:
                    rm3_scores += self.index[f"{field}_snowball"].array.score(term) * weight

        top_k = np.argsort(-rm3_scores)[:k]
        scores = rm3_scores[top_k]
        return top_k, scores
