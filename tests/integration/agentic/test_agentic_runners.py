"""Agentic runner integration tests.

See docs/runner_tests_prd.md for requirements.
"""

import os
import socket
import subprocess

import pytest

from exps.runners.run import RunParams, run_benchmark


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return True


def _bash_service_available() -> bool:
    port = int(os.environ.get("EXPS_BASH_PORT", "8000"))
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def test_run_benchmark_agentic_guarded():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    params = RunParams(
        strategy_path="configs/agentic.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
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


def test_run_benchmark_agentic_wands_bm25_e5_few_shot_delegate():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    params = RunParams(
        strategy_path="configs/agentic_wands_bm25_e5_few_shot_delegate.yml",
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


def test_run_benchmark_agentic_filesystem_tools():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    params = RunParams(
        strategy_path="configs/agentic_filesystem.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert result.summary["tool_calls_mean"] >= 0.0


def test_run_benchmark_agentic_filesystem_traces(tmp_path, monkeypatch):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    trace_root = tmp_path / "search-experiments"
    monkeypatch.setattr("exps.paths.SEARCH_EXPERIMENTS_ROOT", trace_root)
    monkeypatch.setattr("exps.run_dirs.SEARCH_EXPERIMENTS_ROOT", trace_root)

    params = RunParams(
        strategy_path="configs/agentic_filesystem.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty

    trace_base = trace_root / "agentic" / "doug_blog" / "agentic_filesystem_fixture"
    assert trace_base.exists()
    run_dirs = sorted([path for path in trace_base.iterdir() if path.is_dir()])
    assert run_dirs
    query_dirs = [path for path in run_dirs[-1].iterdir() if path.is_dir()]
    assert query_dirs
    log_files = list(query_dirs[0].glob("*.log"))
    assert log_files


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
    reasoning: low
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
        dataset="doug_blog",
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
        dataset="doug_blog",
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
    reasoning: low
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
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty


def test_run_benchmark_agentic_orchestrate_bm25(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    config_path = tmp_path / "agentic_orchestrate.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_orchestrate_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      You take user search queries and orchestrate subagents to find relevant products.
    subagent_system_prompt: |
      You help with tasks searchinging / finding content as instructed.
    search_tools:
      - delegate_task
      - bm25
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty
    assert result.summary["tool_calls_mean"] >= 0.0


def test_run_benchmark_agentic_plan_agents(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    config_path = tmp_path / "agentic_plan.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_plan_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    agents:
      planning:
        system_prompt: |
          You plan how to search for relevant products.
        search_tools:
          - delegate_task
          - bm25
      search:
        system_prompt: |
          You find relevant products and return ranked DOC IDs.
        search_tools:
          - bm25
    plan:
      - planning: plan how to best search for {query}
      - search: find the most relevant results for {query}
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty


def test_run_benchmark_agentic_bash_tool(tmp_path, monkeypatch):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")
    if not _docker_available():
        pytest.skip("Docker is required for bash tool integration test.")
    if not _bash_service_available():
        pytest.skip("Bash service is not running for bash tool integration test.")

    trace_root = tmp_path / "search-experiments"
    monkeypatch.setattr("exps.paths.SEARCH_EXPERIMENTS_ROOT", trace_root)
    monkeypatch.setattr("exps.run_dirs.SEARCH_EXPERIMENTS_ROOT", trace_root)
    monkeypatch.setattr("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", trace_root)

    config_path = tmp_path / "agentic_bash.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_bash_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use the bash tool to search /corpus for relevant products.
    search_tools:
      - bash
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty


def test_agentic_raw_tool_rejected(tmp_path):
    config_path = tmp_path / "agentic_raw_tool.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_raw_tool_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use the search tool to find products.
    search_tools:
      - get_corpus
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="raw search tool"):
        run_benchmark(params)


def test_agentic_dataset_specific_tool_rejected(tmp_path):
    config_path = tmp_path / "agentic_wands_tool.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_wands_tool_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use the search tool to find products.
    search_tools:
      - bm25_wands
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="only available for wands dataset"):
        run_benchmark(params)


def test_agentic_few_shot_happy_path(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    config_path = tmp_path / "agentic_few_shot.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_few_shot_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    few_shot:
      - sample_judgments:
          num_rows: 4
    search_tools:
      - bm25
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert not result.metric_series.empty


def test_agentic_few_shot_missing_column_raises(tmp_path):
    config_path = tmp_path / "agentic_few_shot_bad_col.yml"
    config_path.write_text(
        """
strategy:
  name: agentic_few_shot_bad_col_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    few_shot:
      - sample_judgments:
          num_rows: 4
          columns:
            - missing_col
    search_tools:
      - bm25
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="few_shot column not found"):
        run_benchmark(params)


def test_agentic_codegen_tool_dependency_mismatch(tmp_path):
    reranker_dir = tmp_path / "codegen_dependency_mismatch"
    reranker_dir.mkdir()
    reranker_path = reranker_dir / "reranker.py"
    reranker_path.write_text(
        """
def rerank_doug_blog(query, fielded_bm25, **kwargs):
    docs = fielded_bm25(
        query,
        fields=['title^9.3', 'description^4.1'],
        operator='or',
        top_k=5,
    )
    return [doc['id'] for doc in docs]
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentic_codegen_dep.yml"
    config_path.write_text(
        f"""
strategy:
  name: agentic_codegen_dep_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    search_tools:
      - codegen:
          path: {reranker_dir}
          name: search
          dependencies:
            - bm25
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="codegen tool missing dependencies"):
        run_benchmark(params)


def test_agentic_codegen_tool_return_fields_validation(tmp_path):
    reranker_dir = tmp_path / "codegen_return_fields"
    reranker_dir.mkdir()
    reranker_path = reranker_dir / "reranker.py"
    reranker_path.write_text(
        """
def rerank_doug_blog(query, bm25, **kwargs):
    docs = bm25(query, top_k=5)
    return [doc['id'] for doc in docs]
""".lstrip(),
        encoding="utf-8",
    )
    config_path = tmp_path / "agentic_codegen_return_fields.yml"
    config_path.write_text(
        f"""
strategy:
  name: agentic_codegen_return_fields_fixture
  type: agentic
  params:
    model: gpt-5-mini
    reasoning: low
    system_prompt: |
      Use search tools to find products.
    search_tools:
      - codegen:
          path: {reranker_dir}
          name: search
          dependencies:
            - bm25
          return_fields:
            - missing_col
""".lstrip(),
        encoding="utf-8",
    )
    params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    with pytest.raises(ValueError, match="return_fields not found in corpus"):
        run_benchmark(params)
