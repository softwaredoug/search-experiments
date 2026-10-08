from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity

from exps.bag_of_decisions.decision_generator import DecisionGenerator
from exps.bag_of_decisions.decision_reranker import (
    DecisionReranker,
    ScoredCandidate,
)


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _parse_fields(fields: Any) -> dict[str, float]:
    if not isinstance(fields, (list, tuple)):
        raise ValueError("retrieval_engine.params.fields must be a list of fields.")
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
        if not math.isfinite(parsed_weight) or parsed_weight <= 0:
            raise ValueError(f"BM25 field weight must be positive: {field_spec!r}")
        if field in parsed:
            raise ValueError(f"Duplicate BM25 field: {field}")
        parsed[field] = parsed_weight
    if not parsed:
        raise ValueError("retrieval_engine.params.fields must not be empty.")
    return parsed


class BagOfDecisionsStrategy(SearchStrategy):
    """BM25 retrieval followed by query-generated Jev decision reranking."""

    _type = "bag_of_decisions"

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        no_cache: bool = False,
        **kwargs,
    ):
        return cls(corpus, workers=workers, no_cache=no_cache, **params)

    def __init__(
        self,
        corpus,
        decision_engine: dict,
        retrieval_engine: dict,
        workers: int = 1,
        top_k: int = 10,
        no_cache: bool = False,
        **_unused,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.corpus = corpus
        self.no_cache = no_cache

        if not isinstance(decision_engine, dict):
            raise ValueError("decision_engine must be a mapping.")
        candidate_k = decision_engine.get("k", 100)
        if (
            isinstance(candidate_k, bool)
            or not isinstance(candidate_k, int)
            or candidate_k < 1
        ):
            raise ValueError("decision_engine.k must be a positive integer.")
        self.candidate_k = candidate_k

        if not isinstance(retrieval_engine, dict):
            raise ValueError("retrieval_engine must be a mapping.")
        retrieval_base = retrieval_engine.get("base", "bm25_boosted")
        supported_bases = {
            "bm25_boosted",
            "bm25_filtered",
            "bm25_hierarchy_boosted",
        }
        if retrieval_base not in supported_bases:
            raise ValueError(
                "retrieval_engine.base must be one of: "
                + ", ".join(sorted(supported_bases))
                + "."
            )
        retrieval_params = retrieval_engine.get("params") or {}
        if not isinstance(retrieval_params, dict):
            raise ValueError("retrieval_engine.params must be a mapping.")
        self.fields = _parse_fields(retrieval_params.get("fields") or [])
        for field in self.fields:
            if field not in corpus.columns:
                raise ValueError(f"Missing BM25 field: {field}")
            index_name = f"{field}_snowball"
            if index_name not in corpus:
                corpus[index_name] = SearchArray.index(
                    corpus[field].fillna("").astype(str), snowball_tokenizer
                )
        self.k1 = _finite_number(retrieval_params.get("k1", 1.2), "k1")
        self.b = _finite_number(retrieval_params.get("b", 0.75), "b")
        if self.k1 <= 0:
            raise ValueError("retrieval_engine.params.k1 must be positive.")
        if not 0 <= self.b <= 1:
            raise ValueError("retrieval_engine.params.b must be between 0 and 1.")

        self.decision_generator = DecisionGenerator(
            system_prompt=decision_engine.get("system_prompt"),
            prompt=decision_engine.get("prompt"),
            model=decision_engine.get("model"),
            reasoning=decision_engine.get("reasoning"),
            temperature=decision_engine.get("temperature"),
            verbosity=decision_engine.get("verbosity"),
            no_cache=no_cache,
        )
        self.decision_reranker = DecisionReranker(
            decision_model=decision_engine.get("decision_model"),
            decision_weight=decision_engine.get("decision_weight", 10),
            confidence_threshold=decision_engine.get("confidence_threshold", 0.7),
            state_format=decision_engine.get("state_format"),
            corpus_fields=self.corpus.columns,
            no_cache=no_cache,
        )

    def _baseline_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        similarity = bm25_similarity(k1=self.k1, b=self.b)
        for term in snowball_tokenizer(query):
            for field, weight in self.fields.items():
                scores += (
                    self.corpus[f"{field}_snowball"].array.score(
                        term, similarity=similarity
                    )
                    * weight
                )
        return scores

    def search(self, query: str, k: int = 10):
        if k <= 0 or not len(self.corpus):
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        scores = self._baseline_scores(query)
        candidate_indices = np.argsort(-scores, kind="stable")[: self.candidate_k]
        candidates = [
            ScoredCandidate(
                index=int(index),
                score=float(scores[index]),
                document=self.corpus.iloc[index].to_dict(),
            )
            for index in candidate_indices
        ]
        decisions = self.decision_generator.generate(query)
        reranked = self.decision_reranker.rerank(candidates, decisions)
        for candidate in reranked:
            scores[candidate.index] = candidate.score
        result_indices = np.argsort(-scores, kind="stable")[:k]
        return result_indices, scores[result_indices]

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "generator": self.decision_generator.cache_key,
            "reranker": self.decision_reranker.cache_key,
            "candidate_k": self.candidate_k,
            "fields": self.fields,
            "k1": self.k1,
            "b": self.b,
            "top_k": getattr(self, "top_k", None),
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
