from __future__ import annotations

import numpy as np
import pytest

from cheat_at_search.search import ndcgs, run_strategy
from cheat_at_search.strategy import SearchStrategy

from exps.datasets import get_dataset
from exps.strategies.embedding import EmbeddingStrategy


def _passage_text(row) -> str:
    title = row.get("title")
    description = row.get("description")
    title_text = title.strip() if isinstance(title, str) else ""
    description_text = description.strip() if isinstance(description, str) else ""
    if title_text and description_text:
        return f"{title_text}\n\n{description_text}"
    if title_text:
        return title_text
    return description_text


class DirectEmbeddingStrategy(SearchStrategy):
    def __init__(
        self,
        corpus,
        *,
        model_name: str,
        device: str | None = None,
        top_k: int = 10,
    ):
        super().__init__(corpus, top_k=top_k)
        self.corpus = corpus
        from sentence_transformers import SentenceTransformer

        self._model = (
            SentenceTransformer(model_name, device=device)
            if device
            else SentenceTransformer(model_name)
        )
        texts = [_passage_text(row) for _, row in corpus.iterrows()]
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        embeddings = np.asarray(embeddings)
        norms = np.linalg.norm(embeddings, axis=1)
        norms[norms == 0] = 1.0
        self._doc_embeddings = embeddings / norms[:, None]

    def search(self, query: str, k: int = 10):
        query_emb = self._model.encode([query], convert_to_numpy=True)
        query_emb = np.asarray(query_emb)
        q_norm = np.linalg.norm(query_emb, axis=1)
        q_norm[q_norm == 0] = 1.0
        query_emb = query_emb / q_norm[:, None]
        scores = (query_emb @ self._doc_embeddings.T).reshape(-1)
        top_k = np.argsort(scores)[-k:][::-1]
        return top_k, scores[top_k]


def test_embedding_strategy_matches_direct_minilm(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")

    dataset = get_dataset("wands", ensure_snowball=False)
    corpus = dataset.corpus
    judgments = dataset.judgments

    queries = judgments[["query", "query_id"]].drop_duplicates().sample(
        n=5, random_state=42
    )
    query_list = queries["query"].tolist()
    judgments = judgments[judgments["query"].isin(query_list)].copy()
    doc_ids = judgments["doc_id"].unique().tolist()
    corpus = corpus[corpus["doc_id"].isin(doc_ids)].reset_index(drop=True)

    monkeypatch.setattr(
        "cheat_at_search.embeddings._cache_root",
        lambda: tmp_path,
    )

    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    direct = DirectEmbeddingStrategy(corpus, model_name=model_name)
    cached = EmbeddingStrategy(
        corpus,
        model_name=model_name,
        query_prefix=None,
        document_prefix=None,
        dataset=None,
    )

    direct_results = run_strategy(direct, judgments, queries=query_list, cache=False)
    cached_results = run_strategy(cached, judgments, queries=query_list, cache=False)

    direct_mean = float(ndcgs(direct_results).mean())
    cached_mean = float(ndcgs(cached_results).mean())
    assert direct_mean == pytest.approx(cached_mean, abs=1e-4)
