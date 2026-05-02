from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exps.strategies.bm25 import BM25Strategy
from exps import tools as tools_mod


class _FakeMinilm:
    def encode(self, text):
        if isinstance(text, list):
            return np.stack([self.encode(item) for item in text])
        if text == "alpha":
            return np.array([1.0, 0.0])
        if text == "beta":
            return np.array([0.99, 0.01])
        return np.array([0.0, 1.0])


def test_guard_disallow_similar_queries(monkeypatch):
    monkeypatch.setattr(tools_mod, "_minilm_guard_model", lambda *_args, **_kwargs: _FakeMinilm())

    agent_state = {}
    params = {"tool_name": "search_bm25_guarded", "query": "alpha"}

    err = tools_mod.guard_disallow_similar_queries(params, agent_state, threshold=0.9)
    assert err is None


def test_guarded_tool_limits_top_k():
    def _tool(query: str, top_k: int, agent_state=None):
        return [{"id": 1, "title": query, "description": "", "score": float(top_k)}]

    guarded = tools_mod.make_guarded_search_tool(_tool)
    err = guarded("chair", top_k=21)
    assert isinstance(err, str)


def test_codegen_tool_returns_schema(tmp_path):
    reranker_path = Path(tmp_path) / "reranker.py"
    reranker_path.write_text(
        """
def rerank_wands(query, **kwargs):
    return [102, 101, 103]
""".lstrip(),
        encoding="utf-8",
    )
    corpus = pd.DataFrame(
        {
            "doc_id": [101, 102, 103],
            "title": ["Alpha", "Beta", "Gamma"],
            "description": ["Desc A", "Desc B", "Desc C"],
            "category": ["cat-a", "cat-b", "cat-c"],
        }
    )
    tools = tools_mod.build_search_tools(
        corpus,
        [
            {
                "codegen": {
                    "path": str(tmp_path),
                    "return_fields": ["category"],
                }
            }
        ],
        dataset_name="wands",
    )
    assert len(tools) == 1
    tool = tools[0]
    assert tool.__name__ == "search"

    results = tool("chair", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert results[0]["id"] == 102
    assert results[1]["id"] == 101
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["category"] == "cat-b"
    assert "title" in results[0]
    assert "description" in results[0]


def test_codegen_tool_missing_return_fields_raises(tmp_path):
    reranker_path = Path(tmp_path) / "reranker.py"
    reranker_path.write_text(
        """
def rerank_wands(query, **kwargs):
    return [101]
""".lstrip(),
        encoding="utf-8",
    )
    corpus = pd.DataFrame(
        {
            "doc_id": [101],
            "title": ["Alpha"],
            "description": ["Desc A"],
        }
    )
    with pytest.raises(ValueError, match="return_fields"):
        tools_mod.build_search_tools(
            corpus,
            [
                {
                    "codegen": {
                        "path": str(tmp_path),
                        "return_fields": ["category"],
                    }
                }
            ],
            dataset_name="wands",
        )


def test_fielded_bm25_matches_bm25_strategy():
    corpus = pd.DataFrame(
        {
            "doc_id": [0, 1, 2],
            "title": ["blue chair", "red sofa", "blue couch"],
            "description": [
                "comfortable chair for desk",
                "large sofa with cushions",
                "couch in blue fabric",
            ],
        }
    )
    strategy = BM25Strategy(
        corpus,
        title_boost=9.3,
        description_boost=4.1,
        k1=1.2,
        b=0.75,
        top_k=3,
        workers=1,
    )
    tool = tools_mod.make_fielded_bm25_tool(corpus)
    query = "blue chair"

    indices, scores = strategy.search(query, k=3)
    tool_results = tool(
        keywords=query,
        fields=["title^9.3", "description^4.1"],
        operator="or",
        top_k=3,
        k1=1.2,
        b=0.75,
    )

    tool_ids = [result["id"] for result in tool_results]
    tool_scores = [result["score"] for result in tool_results]
    assert tool_ids == list(indices)
    assert np.allclose(tool_scores, scores)


def test_codegen_tool_missing_dependencies_raises(tmp_path):
    reranker_path = Path(tmp_path) / "reranker.py"
    reranker_path.write_text(
        """
def rerank_wands(query, fielded_bm25, search_embeddings):
    return [101]
""".lstrip(),
        encoding="utf-8",
    )
    corpus = pd.DataFrame(
        {
            "doc_id": [101],
            "title": ["Alpha"],
            "description": ["Desc A"],
        }
    )
    with pytest.raises(ValueError, match="missing dependencies"):
        tools_mod.build_search_tools(
            corpus,
            [
                {
                    "codegen": {
                        "path": str(tmp_path),
                        "dependencies": ["fielded_bm25"],
                    }
                }
            ],
            dataset_name="wands",
        )
