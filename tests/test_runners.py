import os
from types import SimpleNamespace

import pandas as pd
import pytest

from cheat_at_search.search import run_strategy

import exps.agentic
from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.runners.diff import DiffParams, diff_benchmark
from exps.runners.run import RunParams, run_benchmark
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
    assert result.metric_series.index.name == "query_id"
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

    strategy_config = load_strategy_config(params.strategy_path)
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
    if direct_series.index.name != "query_id" and "query_id" in direct_graded.columns:
        query_map = (
            direct_graded[["query", "query_id"]]
            .drop_duplicates()
            .set_index("query")["query_id"]
        )
        direct_series = direct_series.copy()
        direct_series.index = direct_series.index.map(query_map.get)
        direct_series.index.name = "query_id"

    pd.testing.assert_series_equal(result.metric_series, direct_series)


def test_run_benchmark_agentic_guarded():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is required for agentic tests.")

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


def test_agentic_stop_iterations(monkeypatch):
    calls = []

    def fake_agent_run(
        tool_info,
        text_format,
        inputs,
        model="gpt-5-nano",
        agent_state=None,
        summary=True,
        logger=None,
    ):
        calls.append(list(inputs))
        resp = SimpleNamespace(output_parsed="ok", output=[])
        return resp, inputs

    monkeypatch.setattr(exps.agentic, "agent_run", fake_agent_run)

    result = exps.agentic.search(
        tools=[],
        inputs=[{"role": "user", "content": "hi"}],
        stop=[{"iterations": 2}],
    )

    assert result == "ok"
    assert len(calls) == 2


def test_agentic_reprompt_appends(monkeypatch):
    calls = []

    def fake_agent_run(
        tool_info,
        text_format,
        inputs,
        model="gpt-5-nano",
        agent_state=None,
        summary=True,
        logger=None,
    ):
        calls.append(list(inputs))
        resp = SimpleNamespace(output_parsed="ok", output=[])
        return resp, inputs

    monkeypatch.setattr(exps.agentic, "agent_run", fake_agent_run)

    reprompt = "Try harder"
    result = exps.agentic.search(
        tools=[],
        inputs=[{"role": "user", "content": "hi"}],
        stop=[{"iterations": 3}],
        reprompt=reprompt,
    )

    assert result == "ok"
    assert len(calls) == 3
    assert [item["content"] for item in calls[0]] == ["hi"]
    assert [item["content"] for item in calls[1]] == ["hi", reprompt]
    assert [item["content"] for item in calls[2]] == ["hi", reprompt, reprompt]
