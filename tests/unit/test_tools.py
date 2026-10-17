from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from cheat_at_search.agent.sid_legacy_tool import format_tool_schema, tool_schema
from cheat_at_search.agent.pydantize import make_tool_adapter

from exps.strategies.bm25 import BM25Strategy
from exps import tools as tools_mod
from exps.tools.results import format_corpus_result


class _FakeMinilm:
    def encode(self, text):
        if isinstance(text, list):
            return np.stack([self.encode(item) for item in text])
        if text == "alpha":
            return np.array([1.0, 0.0])
        if text == "beta":
            return np.array([0.99, 0.01])
        return np.array([0.0, 1.0])


def test_bm25_tool_annotations_become_openai_parameter_descriptions():
    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["Blue chair"],
            "description": ["A blue desk chair."],
            "category": ["Furniture"],
        }
    )

    _, bm25_spec, _ = make_tool_adapter(tools_mod.make_bm25_tool(corpus))
    bm25_properties = bm25_spec["parameters"]["properties"]
    assert bm25_properties["keywords"]["description"] == "Search query keywords."
    assert "at most 100" in bm25_properties["top_k"]["description"]

    _, fielded_spec, _ = make_tool_adapter(tools_mod.make_fielded_bm25_tool(corpus))
    fielded_properties = fielded_spec["parameters"]["properties"]
    assert "optionally weighted" in fielded_properties["fields"]["description"]
    assert "BM25 k1" in fielded_properties["k1"]["description"]


def test_sid_tool_examples_use_annotated_argument_examples():
    corpus = pd.DataFrame(
        {
            "doc_id": ["product-1"],
            "title": ["Queen platform bed"],
            "description": ["Example bed"],
        }
    )
    tools = tools_mod.build_search_tools(corpus, ["bm25", "read"])

    bm25_example = format_tool_schema(tool_schema(tools[0]))
    read_example = format_tool_schema(tool_schema(tools[1]))

    assert '"keywords": "queen platform bed"' in bm25_example
    assert '"top_k": 10' in bm25_example
    assert '"id": "document_id_from_search"' in read_example
    assert "<tool_call>" in bm25_example
    assert "</tool_call>" in read_example


def test_search_tool_wrappers_preserve_annotated_parameter_descriptions():
    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["Blue chair"],
            "description": ["A blue desk chair."],
            "category": ["Furniture"],
        }
    )
    tool = tools_mod.build_search_tools(
        corpus,
        [
            {
                "bm25": {
                    "columns": [{"source": "category"}],
                    "guards": ["disallow_repeated_queries"],
                }
            }
        ],
    )[0]

    _, tool_spec, _ = make_tool_adapter(tool)
    properties = tool_spec["parameters"]["properties"]
    assert properties["keywords"]["description"] == "Search query keywords."
    assert "at most 100" in properties["top_k"]["description"]


def test_read_tool_returns_a_document_by_doc_id():
    corpus = pd.DataFrame(
        {
            "doc_id": [101, 202],
            "title": ["Red Shoes", "Blue Chair"],
            "description": ["Running shoes.", "A desk chair."],
        }
    )
    read = tools_mod.build_search_tools(corpus, ["read"])[0]

    result = read("202")

    assert read.__name__ == "read"
    assert len(result) == 1
    assert result[0]["id"] == 202
    assert result[0]["title"] == "Blue Chair"
    assert result[0]["description"] == "A desk chair."
    assert read("missing") == []


def test_read_tool_uses_index_as_fallback_document_id():
    corpus = pd.DataFrame(
        {"title": ["Red Shoes"], "description": ["Running shoes."]},
        index=["shoe-doc"],
    )
    read = tools_mod.make_read_tool(corpus)

    result = read("shoe-doc")

    assert result[0]["id"] == "shoe-doc"
    assert result[0]["description"] == "Running shoes."


@patch("exps.tools.guards._minilm_guard_model", new=lambda *_args, **_kwargs: _FakeMinilm())
def test_guard_disallow_similar_queries():
    agent_state = {}
    params = {"tool_name": "search_bm25_guarded", "query": "alpha"}
    err = tools_mod.guard_disallow_similar_queries(params, agent_state, threshold=0.9)
    assert err is None


def test_guarded_tool_limits_top_k():
    def _tool(query: str, top_k: int, agent_state=None):
        return [{"id": 1, "title": query, "description": "", "score": float(top_k)}]

    guarded = tools_mod.make_guarded_search_tool(_tool)
    err = guarded("chair", top_k=101)
    assert isinstance(err, str)


def test_corpus_result_includes_present_image_url_only():
    row = pd.Series(
        {
            "doc_id": 101,
            "title": "Alpha",
            "description": "Desc A",
            "image_url": "https://example.test/alpha.jpg",
        }
    )
    result = format_corpus_result(row)
    assert result["image_url"] == "https://example.test/alpha.jpg"

    without_image = format_corpus_result(
        pd.Series({"doc_id": 102, "title": "Beta", "description": "Desc B"})
    )
    assert "image_url" not in without_image


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
            "image_url": ["https://example.test/a.jpg", "https://example.test/b.jpg", "https://example.test/c.jpg"],
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
    assert results[0]["image_url"] == "https://example.test/b.jpg"


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


def test_fielded_bm25_returns_image_url():
    corpus = pd.DataFrame(
        {
            "doc_id": [0],
            "title": ["blue chair"],
            "description": ["comfortable chair for desk"],
            "image_url": ["https://example.test/chair.jpg"],
        }
    )
    results = tools_mod.make_fielded_bm25_tool(corpus)(
        keywords="blue chair",
        fields=["title", "description"],
        top_k=1,
    )
    assert results[0]["image_url"] == "https://example.test/chair.jpg"


def test_fielded_bm25_phrase_operator():
    corpus = pd.DataFrame(
        {
            "doc_id": [0, 1, 2],
            "title": ["blue chair", "red chair", "chair blue"],
            "description": [
                "blue chair for desk",
                "red chair for desk",
                "chair in blue color",
            ],
        }
    )
    tool = tools_mod.make_fielded_bm25_tool(corpus)
    results = tool(
        keywords="blue chair",
        fields=["title^9.3", "description^4.1"],
        operator="phrase",
        top_k=3,
    )
    assert isinstance(results, list)
    assert len(results) == 3


def test_fielded_bm25_accepts_structured_fields():
    corpus = pd.DataFrame(
        {
            "doc_id": [0, 1],
            "title": ["blue chair", "red sofa"],
            "description": ["chair for desk", "large sofa"],
        }
    )
    tool = tools_mod.make_fielded_bm25_tool(corpus)
    results = tool(
        keywords="blue chair",
        fields=[{"field": "title", "weight": 9.3}, {"description": 4.1}],
        operator="or",
        top_k=2,
    )
    assert isinstance(results, list)
    assert len(results) == 2


def test_fielded_bm25_accepts_unweighted_fields():
    corpus = pd.DataFrame(
        {
            "doc_id": [0, 1],
            "title": ["blue chair", "red sofa"],
            "description": ["chair for desk", "large sofa"],
        }
    )
    tool = tools_mod.make_fielded_bm25_tool(corpus)
    results = tool(
        keywords="blue chair",
        fields=["title"],
        operator="or",
        top_k=2,
    )
    assert isinstance(results, list)
    assert len(results) == 2


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


def test_build_search_tools_rejects_raw_in_agentic_context():
    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["alpha"],
            "description": ["desc"],
        }
    )
    with pytest.raises(ValueError, match="raw search tool"):
        tools_mod.build_search_tools(corpus, ["get_corpus"])


def test_build_search_tools_allows_raw_context():
    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["alpha"],
            "description": ["desc"],
        }
    )
    tools = tools_mod.build_search_tools(corpus, ["get_corpus"], context="raw")
    assert len(tools) == 1
    assert tools[0].__name__ == "get_corpus"
