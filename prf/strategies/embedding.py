from __future__ import annotations

import numpy as np
from cheat_at_search.strategy import SearchStrategy

from prf.embeddings import _minilm_model, load_or_create_embeddings
from prf.mapping import build_doc_id_lookup, doc_ids_to_indices


class EmbeddingStrategy(SearchStrategy):
    """Vector search using SentenceTransformers embeddings."""

    _type = "embedding"

    def __init__(
        self,
        corpus,
        model_name: str,
        top_k: int = 10,
        workers: int = 1,
        device: str | None = None,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.corpus = corpus
        self.model_name = model_name
        self.device = device
        self._lookup = build_doc_id_lookup(corpus)
        self._model = _minilm_model(model_name, device=device)
        self._embeddings = load_or_create_embeddings(
            corpus,
            model_name=model_name,
            device=device,
            show_progress=True,
        )
        doc_norms = np.linalg.norm(self._embeddings, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        self._doc_embeddings = self._embeddings / doc_norms[:, None]

    def _normalize_queries(self, query_embeddings: np.ndarray) -> np.ndarray:
        queries = query_embeddings.astype(float)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        query_norms = np.linalg.norm(queries, axis=1)
        query_norms[query_norms == 0] = 1.0
        return queries / query_norms[:, None]

    def search_batch(self, queries: list[str], k: int = 10):
        if not queries:
            return [], []
        query_embeddings = self._model.encode(queries, convert_to_numpy=True)
        query_embeddings = np.asarray(query_embeddings)
        query_embeddings = self._normalize_queries(query_embeddings)
        scores_matrix = query_embeddings @ self._doc_embeddings.T
        all_top_k = []
        all_scores = []
        for row in scores_matrix:
            top_k = np.argsort(row)[-k:][::-1]
            all_top_k.append(top_k)
            all_scores.append(row[top_k])
        return all_top_k, all_scores

    def search(self, query: str, k: int = 10):
        all_top_k, all_scores = self.search_batch([query], k=k)
        if not all_top_k:
            return [], []
        top_k = all_top_k[0]
        top_scores = all_scores[0]
        if self._lookup:
            indices = doc_ids_to_indices(
                [self.corpus.iloc[idx].get("doc_id", idx) for idx in top_k],
                self._lookup,
            )
            if len(indices) == len(top_k):
                return indices, top_scores
        return top_k, top_scores
