from __future__ import annotations

import json
import numpy as np
from cheat_at_search.strategy import SearchStrategy

from exps.embeddings import _minilm_model, load_or_create_embeddings
from exps.mapping import build_doc_id_lookup, doc_ids_to_indices


class EmbeddingStrategy(SearchStrategy):
    """Vector search using SentenceTransformers embeddings."""

    _type = "embedding"

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        device: str | None = None,
        **kwargs,
    ):
        build_params = dict(params)
        if device and "device" not in build_params:
            build_params["device"] = device
        return cls(corpus, workers=workers, **build_params)

    def __init__(
        self,
        corpus,
        model_name: str,
        top_k: int = 10,
        workers: int = 1,
        device: str | None = None,
        doc_chunk_size: int = 100000,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.corpus = corpus
        self.model_name = model_name
        self.device = device
        self.doc_chunk_size = doc_chunk_size
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

    def _search_batch(self, queries: list[str], k: int = 10):
        if not queries:
            return [], []
        query_embeddings = self._model.encode(queries, convert_to_numpy=True)
        query_embeddings = np.asarray(query_embeddings)
        query_embeddings = self._normalize_queries(query_embeddings)
        num_docs = self._doc_embeddings.shape[0]
        chunk_size = max(int(self.doc_chunk_size), 1)
        num_queries = query_embeddings.shape[0]
        best_scores = np.full((num_queries, k), -np.inf)
        best_indices = np.full((num_queries, k), -1, dtype=int)

        for start in range(0, num_docs, chunk_size):
            end = min(start + chunk_size, num_docs)
            doc_chunk = self._doc_embeddings[start:end]
            scores_chunk = query_embeddings @ doc_chunk.T

            for row_idx in range(num_queries):
                row = scores_chunk[row_idx]
                if row.size == 0:
                    continue
                if row.size <= k:
                    chunk_top_idx = np.arange(row.size)
                else:
                    chunk_top_idx = np.argpartition(row, -k)[-k:]
                chunk_scores = row[chunk_top_idx]
                chunk_indices = chunk_top_idx + start

                combined_scores = np.concatenate([best_scores[row_idx], chunk_scores])
                combined_indices = np.concatenate([best_indices[row_idx], chunk_indices])
                if combined_scores.size <= k:
                    select_idx = np.arange(combined_scores.size)
                else:
                    select_idx = np.argpartition(combined_scores, -k)[-k:]
                best_scores[row_idx] = combined_scores[select_idx]
                best_indices[row_idx] = combined_indices[select_idx]

        all_top_k = []
        all_scores = []
        for row_idx in range(num_queries):
            order = np.argsort(best_scores[row_idx])[::-1]
            all_top_k.append(best_indices[row_idx][order])
            all_scores.append(best_scores[row_idx][order])
        return all_top_k, all_scores

    def search_batch(self, queries: list[str], k: int = 10):
        return self._search_batch(queries, k=k)

    def search(self, query: str, k: int = 10):
        all_top_k, all_scores = self._search_batch([query], k=k)
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

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model_name": self.model_name,
            "device": self.device,
            "doc_chunk_size": self.doc_chunk_size,
            "top_k": getattr(self, "top_k", None),
        }
        return json.dumps(payload, sort_keys=True, default=str)
