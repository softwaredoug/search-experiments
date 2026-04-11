import numpy as np
from searcharray import SearchArray

from cheat_at_search.strategy import SearchStrategy
from searcharray.similarity import compute_idf
from cheat_at_search.tokenizers import snowball_tokenizer
from .bm25_vect import top_n_term_strengths


class PRFRerankStrategy(SearchStrategy):
    def __init__(
        self,
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        rm3_fields=None,
        binary_relevance_fields=None,
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
        self.binary_relevance_fields = self._normalize_binary_relevance_fields(
            binary_relevance_fields
        )
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

    def _normalize_binary_relevance_fields(self, binary_relevance_fields):
        if binary_relevance_fields is None:
            return None
        allowed_fields = {"title", "description", "category"}
        if isinstance(binary_relevance_fields, str):
            binary_relevance_fields = [
                field.strip()
                for field in binary_relevance_fields.split(",")
                if field.strip()
            ]
        if not binary_relevance_fields:
            raise ValueError("Binary relevance fields cannot be empty")
        unknown_fields = [
            field for field in binary_relevance_fields if field not in allowed_fields
        ]
        if unknown_fields:
            unknown_fields_str = ", ".join(sorted(unknown_fields))
            allowed_fields_str = ", ".join(sorted(allowed_fields))
            raise ValueError(
                "Unknown binary relevance fields: "
                f"{unknown_fields_str}. Allowed: {allowed_fields_str}"
            )
        return set(binary_relevance_fields)

    def _prf_rerank_scores(
        self,
        query_terms,
        doc_weights,
        field,
        binary_relevance=True,
        return_vectors=False,
        debug_terms=None,
    ):
        arr = self.index[field].array

        all_terms, exp_vects, exp_top_ns, debug_info = top_n_term_strengths(
            arr,
            doc_weights,
            query_terms=query_terms,
            binary_relevance=binary_relevance,
            mu=1000,
            debug_terms=debug_terms,
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
            return all_top_n_scores, doc_vectors, debug_info
        return all_top_n_scores, debug_info

    def _search(self, query, k=10, return_vectors=False, debug_terms=None):
        tokenized = snowball_tokenizer(query)
        bm25_scores = np.zeros(len(self.index))
        num_matches = np.zeros(len(self.index))
        df_weights = np.zeros(len(self.index))
        for token in tokenized:
            matches = np.zeros(len(self.index), dtype=bool)
            df_title = 0
            df_description = 0
            if "title_snowball" in self.index:
                term_match = self.index["title_snowball"].array.score(token)
                bm25_scores += term_match * self.title_boost
                matches |= term_match > 0
                df_title = self.index["title_snowball"].array.docfreq(token)

            if "description_snowball" in self.index:
                term_match = self.index["description_snowball"].array.score(token)
                bm25_scores += term_match * self.description_boost
                matches |= term_match > 0
                df_description = self.index["description_snowball"].array.docfreq(token)
            df = max(df_title, df_description)
            df_weights[matches] += compute_idf(len(self.index), df)
            num_matches += matches.astype(int)
        doc_weight = bm25_scores.copy()
        doc_weight *= df_weights
        # doc_weight[~all_terms_match] = 0

        if return_vectors:
            doc_vectors = {}
            field_debug_info = {} if debug_terms else None
            for field in self.rm3_fields:
                scores, field_vectors, debug_info = self._prf_rerank_scores(
                    tokenized,
                    doc_weight,
                    f"{field}_snowball",
                    binary_relevance=self._binary_relevance_for_field(field),
                    return_vectors=True,
                    debug_terms=debug_terms,
                )
                bm25_scores += scores
                if field_debug_info is not None:
                    field_debug_info[field] = debug_info
                for doc_id, term_scores in field_vectors.items():
                    doc_vector = doc_vectors.get(doc_id)
                    if doc_vector is None:
                        doc_vectors[doc_id] = term_scores
                        continue
                    for term, score in term_scores.items():
                        doc_vector[term] = doc_vector.get(term, 0.0) + score
        else:
            for field in self.rm3_fields:
                scores, _ = self._prf_rerank_scores(
                    tokenized,
                    doc_weight,
                    f"{field}_snowball",
                    binary_relevance=self._binary_relevance_for_field(field),
                    debug_terms=debug_terms,
                )
                bm25_scores += scores

        top_k = np.argsort(-bm25_scores)[:k]
        scores = bm25_scores[top_k]
        if return_vectors:
            if debug_terms:
                return doc_vectors, top_k, scores, field_debug_info
            return doc_vectors, top_k, scores
        return top_k, scores

    def search(self, query, k=10):
        return self._search(query, k=k)

    def vectors(self, query, k=10, debug_terms=None):
        return self._search(query, k=k, return_vectors=True, debug_terms=debug_terms)

    def _binary_relevance_for_field(self, field):
        if self.binary_relevance_fields is None:
            return field != "description"
        return field in self.binary_relevance_fields

        # Doc these two together
