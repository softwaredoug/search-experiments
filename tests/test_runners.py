"""Runner integration tests.

See docs/runner_tests_prd.md for requirements.
"""

import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cheat_at_search.search import run_strategy

import exps.agentic
from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.runners.diff import DiffParams, diff_benchmark
from exps.runners.run import RunParams, run_benchmark
from exps.runners.train import TrainParams, train_strategy
from exps.strategy_config import load_strategy_config, resolve_strategy_class


def test_run_benchmark_wands_bm25_all_params():
    params = RunParams(
        strategy_path="configs/bm25.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        binary_relevance="title",
        device=None,
    )
    result = run_benchmark(params)

    assert result.strategy_name == "bm25_fixture"
    assert isinstance(result.metric_series, pd.Series)
    assert result.metric_series.index.name == "query"
    assert not result.metric_series.empty
    assert "mean_" in next(iter(result.summary.keys()))
    assert result.summary["tool_calls_mean"] == 1.0
    assert result.summary["tool_calls_median"] == 1.0
    assert result.summary["tool_calls_std"] == 0.0


def test_diff_benchmark_wands_bm25_all_params():
    params = DiffParams(
        strategy_a_path="configs/bm25.yml",
        strategy_b_path="configs/bm25.yml",
        base_path="tests/fixtures",
        dataset="wands",
        query=None,
        k=5,
        num_queries=2,
        seed=123,
        workers=1,
        sort="delta",
        binary_relevance="title",
        device=None,
    )
    result = diff_benchmark(params)

    assert isinstance(result.metric_a, pd.Series)
    assert isinstance(result.metric_b, pd.Series)
    assert "diff" in result.diff_table.columns


def test_diff_benchmark_wands_query_results():
    params = DiffParams(
        strategy_a_path="configs/bm25.yml",
        strategy_b_path="configs/bm25.yml",
        base_path="tests/fixtures",
        dataset="wands",
        query="salon chair",
        k=5,
        num_queries=1,
        seed=123,
        workers=1,
        sort="delta",
        binary_relevance="title",
        device=None,
    )
    result = diff_benchmark(params)

    assert result.query_results_a is not None
    assert result.query_results_b is not None
    assert len(result.query_results_a) == 5
    assert len(result.query_results_b) == 5


def test_run_benchmark_query_results():
    params = RunParams(
        strategy_path="configs/bm25.yml",
        base_path="tests/fixtures",
        dataset="wands",
        query="salon chair",
        k=5,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.query_results is not None
    assert len(result.query_results) == 5
    assert "score" in result.query_results.columns
    assert "display_title" in result.query_results.columns
    assert result.most_relevant_row is not None
    assert result.most_relevant_grade_col is not None
    assert result.relevant_examples is not None
    assert len(result.relevant_examples) <= 3
    for example in result.relevant_examples:
        assert "doc_id" in example
        assert "title" in example
        assert "description" in example


def test_run_benchmark_matches_direct():
    params = RunParams(
        strategy_path="configs/bm25.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        binary_relevance=None,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    strategy_config = load_strategy_config(params.strategy_path, base_path=params.base_path)
    strategy_cls = resolve_strategy_class(strategy_config.type)
    strategy_params = dict(strategy_config.params)
    dataset = get_dataset(params.dataset)
    corpus = dataset.corpus
    judgments = dataset.judgments
    strategy = strategy_cls(corpus, workers=params.workers, **strategy_params)
    available_queries = judgments[["query", "query_id"]].drop_duplicates()
    available_queries = available_queries.sample(params.num_queries, random_state=params.seed)
    queries = available_queries["query"].tolist()
    direct_graded = run_strategy(
        strategy,
        judgments,
        queries=queries,
        seed=params.seed,
        cache=not params.no_cache,
    )
    metric_name, metric_fn = metric_for_dataset(params.dataset)
    direct_series = metric_fn(direct_graded)
    pd.testing.assert_series_equal(result.metric_series, direct_series)


def test_run_benchmark_agentic_guarded():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    params = RunParams(
        strategy_path="configs/agentic.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)
    assert not result.metric_series.empty
    assert result.summary["tool_calls_mean"] >= 0.0
    assert result.summary["tool_calls_median"] >= 0.0
    assert result.summary["tool_calls_std"] >= 0.0


def test_run_benchmark_agentic_codegen_tool(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    codegen_dir = tmp_path / "codegen_run"
    codegen_dir.mkdir()
    reranker_path = codegen_dir / "reranker.py"
    reranker_path.write_text(
        """
def rerank_wands(query, fielded_bm25, **kwargs):
    docs = fielded_bm25(
        keywords=query,
        fields=['title^9.3', 'description^4.1'],
        operator='or',
        top_k=10,
    )
    return [doc['id'] for doc in docs]
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentic_codegen.yml"
    config_path.write_text(
        f"""
strategy:
  name: agentic_codegen_fixture
  type: agentic
  params:
    model: gpt-5-mini
    system_prompt: |
      You take user search queries and use search tools to find the most relevant products.
    search_tools:
      - codegen:
          path: {codegen_dir}
          name: search
          dependencies:
            - fielded_bm25
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert not result.metric_series.empty
    assert result.summary["tool_calls_mean"] >= 0.0


def test_run_benchmark_agentic_codegen_fixture_nonzero():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    params = RunParams(
        strategy_path="configs/agentic_w_codegen.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert (result.metric_series > 0).any()


def test_run_benchmark_agentic_query_rewrite_tool(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    config_path = tmp_path / "agentic_query_rewrite.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_query_rewrite_fixture
  type: agentic
  params:
    model: gpt-5-mini
    system_prompt: |
      You take user search queries and use search tools to find the most relevant products.
    search_tools:
      - query_rewrite:
          model: gpt-5-mini
          max_alternatives: 2
      - bm25
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty


def test_run_benchmark_codegen_guarded():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    params = RunParams(
        strategy_path="configs/codegen_guarded.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty


def test_train_codegen_guarded():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    params = TrainParams(
        strategy_path="configs/codegen_guarded.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path


def test_agentic_stop_iterations():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    reprompt = "Please try again with SearchResultsIds."
    inputs = [{"role": "user", "content": "Return SearchResultsIds for any 10 ids."}]
    result = exps.agentic.search(
        tools=[],
        inputs=inputs,
        stop=[{"iterations": 2}],
        reprompt=reprompt,
        model="gpt-5-nano",
    )

    assert isinstance(result, exps.agentic.SearchResultsIds)
    reprompt_count = sum(
        1
        for item in inputs
        if isinstance(item, dict)
        and item.get("role") == "user"
        and item.get("content") == reprompt
    )
    assert reprompt_count == 1


def test_agentic_stop_tool_calls():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    def echo_tool(message: str, agent_state=None) -> dict[str, str]:
        """Return the provided message."""
        return {"message": message}

    agent_state = {"num_tool_calls": 0}
    inputs = [
        {
            "role": "user",
            "content": (
                "Call the echo_tool twice with message 'ping-1' and 'ping-2'. "
                "Then respond with SearchResultsIds."
            ),
        }
    ]
    result = exps.agentic.search(
        tools=[echo_tool],
        inputs=inputs,
        stop=[{"tool_calls": 2}, {"iterations": 3}],
        reprompt="Remember to call echo_tool before answering.",
        agent_state=agent_state,
        model="gpt-5-nano",
    )

    assert isinstance(result, exps.agentic.SearchResultsIds)
    assert agent_state["num_tool_calls"] >= 2


def test_agentic_reprompt_appends():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    reprompt = "Try again with a new ordering."
    inputs = [{"role": "user", "content": "Return SearchResultsIds for any 10 ids."}]
    result = exps.agentic.search(
        tools=[],
        inputs=inputs,
        stop=[{"iterations": 3}],
        reprompt=reprompt,
        model="gpt-5-nano",
    )

    assert isinstance(result, exps.agentic.SearchResultsIds)
    reprompt_count = sum(
        1
        for item in inputs
        if isinstance(item, dict)
        and item.get("role") == "user"
        and item.get("content") == reprompt
    )
    assert reprompt_count == 2


def test_run_benchmark_embedding_prefixes(monkeypatch, tmp_path):
    def fake_cache_root():
        return tmp_path

    monkeypatch.setattr("cheat_at_search.embeddings._cache_root", fake_cache_root)

    corpus = pd.DataFrame(
        {
            "doc_id": list(range(10)),
            "title": [f"Doc {i}" for i in range(10)],
            "description": [f"Description {i}" for i in range(10)],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": [1],
            "query": ["blue jeans"],
            "doc_id": [0],
            "grade": [1],
        }
    )

    dataset = SimpleNamespace(corpus=corpus, judgments=judgments)
    monkeypatch.setattr("exps.runners.run.get_dataset", lambda *args, **kwargs: dataset)

    params = RunParams(
        strategy_path="configs/embedding_e5_base_v2.yml",
        base_path="tests/fixtures",
        dataset="wands",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert len(result.metric_series) == 1
