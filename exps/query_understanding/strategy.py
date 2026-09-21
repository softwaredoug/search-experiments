from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity

from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from exps.query_understanding.enrichers import Enricher, make_enricher

MAX_CATEGORY_CARDINALITY = 300


def _parse_fields(fields: list[str]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for field_spec in fields:
        if not isinstance(field_spec, str) or not field_spec.strip():
            raise ValueError("retrieval_engine.params.fields must contain strings.")
        field, separator, weight = field_spec.rpartition("^")
        if not separator:
            field, weight = field_spec, "1.0"
        field = field.strip()
        if not field:
            raise ValueError(f"Invalid BM25 field specification: {field_spec!r}")
        try:
            parsed_weight = float(weight)
        except ValueError as exc:
            raise ValueError(f"Invalid BM25 field weight: {field_spec!r}") from exc
        if not np.isfinite(parsed_weight) or parsed_weight <= 0:
            raise ValueError(f"BM25 field weight must be positive: {field_spec!r}")
        if field in parsed:
            raise ValueError(f"Duplicate BM25 field: {field}")
        parsed[field] = parsed_weight
    if not parsed:
        raise ValueError("retrieval_engine.params.fields must not be empty.")
    return parsed


class QueryUnderstandingStrategy(SearchStrategy):
    _type = "query_understanding"

    @classmethod
    def build(cls, params: dict, *, corpus, workers: int = 1, **kwargs):
        categorize = params.get("categorize") or {}
        category_field = categorize.get("field")
        if not isinstance(category_field, str) or not category_field:
            raise ValueError("categorize.field must be a non-empty string.")
        if category_field not in corpus.columns:
            raise ValueError(f"Missing category field: {category_field}")
        values = corpus[category_field].dropna().astype(str)
        values = values[values != ""]
        category_counts = values.value_counts()
        if len(category_counts) > MAX_CATEGORY_CARDINALITY:
            warnings.warn(
                f"Category field {category_field!r} has {len(category_counts)} values; "
                f"using the top {MAX_CATEGORY_CARDINALITY} by corpus frequency.",
                UserWarning,
                stacklevel=2,
            )
        vocabulary = category_counts.head(MAX_CATEGORY_CARDINALITY).index.tolist()
        enricher = make_enricher(
            categorize.get("enrichment_engine"),
            field=category_field,
            vocabulary=vocabulary,
            model=params.get("model", "gpt-5-mini"),
            reasoning=params.get("reasoning"),
        )
        return cls(corpus, workers=workers, enricher=enricher, **params)

    def __init__(
        self,
        corpus,
        categorize: dict,
        retrieval_engine: dict,
        enricher: Enricher,
        workers: int = 1,
        top_k: int = 10,
        **_unused,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.index = corpus
        self.category_field = categorize.get("field")
        if not isinstance(self.category_field, str) or not self.category_field:
            raise ValueError("categorize.field must be a non-empty string.")
        if self.category_field not in corpus.columns:
            raise ValueError(f"Missing category field: {self.category_field}")
        if not hasattr(enricher, "enrich") or not hasattr(enricher, "cache_key"):
            raise TypeError("enricher must implement the Enricher protocol.")
        self.enricher = enricher

        retrieval_config = retrieval_engine or {}
        self.retrieval_base = retrieval_config.get("base", "bm25_boosted")
        if self.retrieval_base not in {"bm25_boosted", "bm25_filtered"}:
            raise ValueError(
                "retrieval_engine.base must be bm25_boosted or bm25_filtered."
            )
        retrieval_params = retrieval_config.get("params") or {}
        self.fields = _parse_fields(retrieval_params.get("fields") or [])
        self.category_boost = float(retrieval_params.get("boost_matches", 0.0))
        if not np.isfinite(self.category_boost) or self.category_boost < 0:
            raise ValueError("boost_matches must be a non-negative number.")

        for field in self.fields:
            if field not in corpus.columns:
                raise ValueError(f"Missing BM25 field: {field}")
            index_name = f"{field}_snowball"
            if index_name not in self.index:
                self.index[index_name] = SearchArray.index(
                    corpus[field].fillna(""), snowball_tokenizer
                )
        self.category_index_name = f"{self.category_field}_snowball"
        if self.category_index_name not in self.index:
            self.index[self.category_index_name] = SearchArray.index(
                corpus[self.category_field].fillna(""), snowball_tokenizer
            )
        self.k1 = float(retrieval_params.get("k1", 1.2))
        self.b = float(retrieval_params.get("b", 0.75))

    def _category_matches(self, categories: list[str]) -> np.ndarray:
        matches = np.zeros(len(self.index), dtype=bool)
        for category in categories:
            terms = snowball_tokenizer(category)
            if terms:
                matches |= self.index[self.category_index_name].array.score(terms) > 0
        return matches

    def enrich(self, query: str) -> list[str]:
        return self.enricher.enrich(query)

    def search(self, query: str, k: int = 10):
        query_terms = snowball_tokenizer(query)
        scores = np.zeros(len(self.index), dtype=float)
        similarity = bm25_similarity(k1=self.k1, b=self.b)
        for term in query_terms:
            for field, weight in self.fields.items():
                scores += (
                    self.index[f"{field}_snowball"].array.score(
                        term, similarity=similarity
                    )
                    * weight
                )

        categories = self.enrich(query)
        category_matches = self._category_matches(categories)
        if self.retrieval_base == "bm25_filtered" and categories:
            scores = np.where(category_matches, scores, -np.inf)
        elif self.retrieval_base == "bm25_boosted":
            scores[category_matches] += self.category_boost

        top_indices = np.argsort(-scores)[:k]
        return top_indices, scores[top_indices]

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "category_field": self.category_field,
            "retrieval_base": self.retrieval_base,
            "fields": self.fields,
            "category_boost": self.category_boost,
            "k1": self.k1,
            "b": self.b,
            "top_k": getattr(self, "top_k", None),
            "enricher": self.enricher.cache_key,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
