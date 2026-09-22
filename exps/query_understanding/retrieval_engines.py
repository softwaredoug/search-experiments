from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from cheat_at_search.tokenizers import snowball_tokenizer


class RetrievalEngine(ABC):
    """Apply query-understanding classifications to baseline search scores."""

    def __init__(self, index, category_index_name: str, params: dict):
        self.index = index
        self.category_index_name = category_index_name
        self.params = params

    def category_matches(self, category: str) -> np.ndarray:
        terms = snowball_tokenizer(category)
        if not terms:
            return np.zeros(len(self.index), dtype=bool)
        return self.index[self.category_index_name].array.score(terms) > 0

    @abstractmethod
    def apply(self, scores: np.ndarray, categories: list[str]) -> np.ndarray:
        raise NotImplementedError


class BM25BoostedRetrievalEngine(RetrievalEngine):
    def apply(self, scores: np.ndarray, categories: list[str]) -> np.ndarray:
        boosted_scores = scores.copy()
        boost = self.params["boost_matches"]
        for category in categories:
            boosted_scores[self.category_matches(category)] += boost
        return boosted_scores


class BM25FilteredRetrievalEngine(RetrievalEngine):
    def apply(self, scores: np.ndarray, categories: list[str]) -> np.ndarray:
        if not categories:
            return scores
        matches = np.zeros(len(self.index), dtype=bool)
        for category in categories:
            matches |= self.category_matches(category)
        return np.where(matches, scores, -np.inf)


class BM25HierarchyBoostedRetrievalEngine(RetrievalEngine):
    def apply(self, scores: np.ndarray, categories: list[str]) -> np.ndarray:
        boosted_scores = scores.copy()
        boost = self.params["boost_matches"]
        decay = self.params["decay"]
        for category in categories:
            parts = [part.strip() for part in category.split("/") if part.strip()]
            levels = [
                " / ".join(parts[:index]) for index in range(1, len(parts) + 1)
            ]
            for level, level_category in enumerate(levels):
                boosted_scores[self.category_matches(level_category)] += boost * decay**level
        return boosted_scores


def make_retrieval_engine(
    base: str, *, index, category_index_name: str, params: dict
) -> RetrievalEngine:
    engine_types = {
        "bm25_boosted": BM25BoostedRetrievalEngine,
        "bm25_filtered": BM25FilteredRetrievalEngine,
        "bm25_hierarchy_boosted": BM25HierarchyBoostedRetrievalEngine,
    }
    try:
        engine_type = engine_types[base]
    except KeyError as exc:
        supported = ", ".join(engine_types)
        raise ValueError(f"retrieval_engine.base must be one of: {supported}.") from exc
    return engine_type(index, category_index_name, params)
