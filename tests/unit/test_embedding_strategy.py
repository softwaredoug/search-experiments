from __future__ import annotations

import numpy as np
import pandas as pd
from cheat_at_search.embeddings import NumpyArrayIterator
from unittest.mock import patch

from exps.strategies.embedding import EmbeddingStrategy


class _FakeEmbeddingModel:
    def encode(self, queries, convert_to_numpy=True):
        return np.ones((len(queries), 2), dtype=float)


def test_embedding_strategy_handles_chunked_embeddings(tmp_path):
    chunk_path = tmp_path / "embeddings_chunk_0.npy"
    np.save(chunk_path, np.array([[1.0, 0.0], [0.0, 1.0]]))
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "title": ["Alpha", "Beta"],
            "description": ["First", "Second"],
        }
    )

    with patch(
        "exps.strategies.embedding.load_or_create_embeddings",
        return_value=(NumpyArrayIterator([str(chunk_path)]), _FakeEmbeddingModel()),
    ):
        strategy = EmbeddingStrategy(
            corpus,
            model_name="test-model",
            query_prefix=None,
            document_prefix=None,
        )

    indices, scores = strategy.search("query", k=2)

    assert list(indices) == [1, 0]
    np.testing.assert_allclose(scores, [1 / np.sqrt(2), 1 / np.sqrt(2)])
