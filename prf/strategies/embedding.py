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
        )

    def _cosine_similarity(self, query_embedding: np.ndarray) -> np.ndarray:
        query = query_embedding.astype(float)
        query_norm = np.linalg.norm(query) or 1.0
        doc_norms = np.linalg.norm(self._embeddings, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        return np.dot(self._embeddings, query) / (doc_norms * query_norm)

    def search(self, query: str, k: int = 10):
        query_embedded = self._model.encode(query, convert_to_numpy=True)
        if query_embedded.ndim != 1:
            query_embedded = np.asarray(query_embedded).reshape(-1)
        scores = self._cosine_similarity(query_embedded)
        top_k = np.argsort(scores)[-k:][::-1]
        top_scores = scores[top_k]
        if self._lookup:
            indices = doc_ids_to_indices(
                [self.corpus.iloc[idx].get("doc_id", idx) for idx in top_k],
                self._lookup,
            )
            if len(indices) == len(top_k):
                return indices, top_scores
        return top_k, top_scores
