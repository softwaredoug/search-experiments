from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from searcharray import SearchArray
from cheat_at_search.agent.sid_legacy_tool import format_tool_schema, tool_schema
from cheat_at_search.agent.pydantize import make_tool_adapter
from cheat_at_search.agent.sid_tools import build_sid2_tool_config
from cheat_at_search.tokenizers import snowball_tokenizer

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
    assert bm25_properties["query"]["description"] == "Search query keywords."
    assert "at most 100" in bm25_properties["limit"]["description"]
    assert "keywords" not in bm25_properties
    assert "top_k" not in bm25_properties

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

    assert '"query": "queen platform bed"' in bm25_example
    assert '"limit": 10' in bm25_example
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
    corpus["title_snowball"] = SearchArray.index(corpus["title"], snowball_tokenizer)
    corpus["description_snowball"] = SearchArray.index(
        corpus["description"], snowball_tokenizer
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
    assert tool.aliases == ["text_search"]
    assert properties["query"]["description"] == "Search query keywords."
    assert "at most 100" in properties["limit"]["description"]
    assert "keywords" not in properties
    assert "top_k" not in properties

    # Old generated callers remain executable, while the exposed schema stays canonical.
    assert tool(keywords="blue chair", top_k=1)[0]["id"] == 1


def test_search_tool_families_receive_sid2_aliases(monkeypatch):
    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["Blue chair"],
            "description": ["A blue desk chair."],
        }
    )
    text_tools = tools_mod.build_search_tools(corpus, ["bm25", "fielded_bm25"])
    assert [tool.aliases for tool in text_tools] == [
        ["text_search"],
        ["text_search"],
    ]
    sid_specs, sid_examples = build_sid2_tool_config([text_tools[0]])
    sid_properties = sid_specs[0]["parameters"]["properties"]
    assert sid_specs[0]["name"] == "text_search"
    assert "query" in sid_properties and "limit" in sid_properties
    assert "keywords" not in sid_properties and "top_k" not in sid_properties
    assert sid_examples[0]["arguments"] == {
        "query": "queen platform bed",
        "limit": 10,
    }

    monkeypatch.setattr(
        "exps.tools.embeddings.load_or_create_embeddings",
        lambda corpus, **_kwargs: (np.ones((len(corpus), 2)), _FakeMinilm()),
    )
    embedding_tools = tools_mod.build_search_tools(corpus, ["minilm"])
    assert embedding_tools[0].aliases == ["search"]
    _, embedding_spec, _ = make_tool_adapter(embedding_tools[0])
    embedding_properties = embedding_spec["parameters"]["properties"]
    assert "query" in embedding_properties
    assert "limit" in embedding_properties
    assert "question" not in embedding_properties
    assert "top_k" not in embedding_properties
    assert embedding_tools[0](question="alpha", top_k=1)[0]["id"] == 1
    sid_specs, _ = build_sid2_tool_config([embedding_tools[0]])
    sid_properties = sid_specs[0]["parameters"]["properties"]
    assert sid_specs[0]["name"] == "search"
    assert "query" in sid_properties and "limit" in sid_properties
    assert "question" not in sid_properties and "top_k" not in sid_properties

    expected_aliases = {
        "bm25": ["text_search"],
        "fielded_bm25": ["text_search"],
        "bm25_wands": ["text_search"],
        "bm25_wands_prefiltered": ["text_search"],
        "minilm": ["search"],
        "embeddings": ["search"],
        "e5_base_v2": ["search"],
        "minilm_wands": ["search"],
        "e5_base_v2_wands": ["search"],
        "e5_base_v2_wands_prefiltered": ["search"],
    }
    assert {
        name: tools_mod.TOOL_REGISTRY[name]["aliases"] for name in expected_aliases
    } == expected_aliases


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


def test_guarded_tool_limits_result_limit():
    def _tool(query: str, limit: int, agent_state=None):
        return [{"id": 1, "title": query, "description": "", "score": float(limit)}]

    guarded = tools_mod.make_guarded_search_tool(_tool)
    err = guarded("chair", limit=101)
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
    _, tool_spec, _ = make_tool_adapter(tool)
    assert "limit" in tool_spec["parameters"]["properties"]
    assert "top_k" not in tool_spec["parameters"]["properties"]

    results = tool("chair", limit=2)
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
        query=query,
        fields=["title^9.3", "description^4.1"],
        operator="or",
        limit=3,
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
        query="blue chair",
        fields=["title", "description"],
        limit=1,
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
        query="blue chair",
        fields=["title^9.3", "description^4.1"],
        operator="phrase",
        limit=3,
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
        query="blue chair",
        fields=[{"field": "title", "weight": 9.3}, {"description": 4.1}],
        operator="or",
        limit=2,
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
        query="blue chair",
        fields=["title"],
        operator="or",
        limit=2,
    )
    assert isinstance(results, list)
    assert len(results) == 2


@patch("exps.tools.wands.load_or_create_embeddings")
@patch("exps.tools.wands.load_model")
def test_wands_embedding_tool_accepts_query_without_prefix(
    mock_load_model,
    mock_load_or_create_embeddings,
):
    corpus = pd.DataFrame(
        {
            "doc_id": [1],
            "title": ["Blue chair"],
            "description": ["A blue desk chair."],
            "category": ["Furniture"],
        }
    )
    model = _FakeMinilm()
    mock_load_or_create_embeddings.return_value = (
        np.array([[0.0, 1.0]]),
        model,
    )
    mock_load_model.return_value = model

    tool = tools_mod.make_wands_embedding_tool(corpus)
    results = tool(query="blue chair", limit=1)

    assert len(results) == 1
    assert results[0]["id"] == 1


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
