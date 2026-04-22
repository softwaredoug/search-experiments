from __future__ import annotations

from cheat_at_search.strategy import SearchStrategy

from prf.mapping import build_doc_id_lookup, doc_ids_to_indices
from prf.tools import make_embedding_tool


class EmbeddingStrategy(SearchStrategy):
    """Naive search over precomputed MiniLM embeddings."""

    def __init__(self, corpus, top_k: int = 10, workers: int = 1):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.corpus = corpus
        self._lookup = build_doc_id_lookup(corpus)
        self._tool = make_embedding_tool(corpus)

        try:
            from cheat_at_search.wands_data import product_embeddings
        except Exception as exc:
            raise ValueError(
                "Embedding strategy is only available for WANDS data."
            ) from exc
        if len(product_embeddings) != len(corpus):
            raise ValueError("Embedding strategy only supports the WANDS corpus.")

    def search(self, query: str, k: int = 10):
        resp = self._tool(query, top_k=k)
        ids = [r["id"] for r in resp]
        scores = [r["score"] for r in resp]
        if self._lookup:
            indices = doc_ids_to_indices(ids, self._lookup)
            scores = scores[: len(indices)]
            return indices, scores
        return ids, scores
