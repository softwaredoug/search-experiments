from __future__ import annotations

import pytest

from exps.strategies.embedding import EmbeddingStrategy


def test_embedding_strategy_with_real_cheat_at_search_embeddings(
    doug_blog_dataset, monkeypatch, tmp_path
):
    pytest.importorskip("sentence_transformers")

    import cheat_at_search.embeddings as embeddings

    monkeypatch.setattr(embeddings, "_cache_root", lambda: tmp_path)
    corpus = doug_blog_dataset.corpus.head(2).reset_index(drop=True)

    strategy = EmbeddingStrategy(
        corpus,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        query_prefix=None,
        document_prefix=None,
    )

    indices, scores = strategy.search("search", k=2)

    assert len(indices) == 2
    assert len(scores) == 2
