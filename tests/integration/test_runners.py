"""Runner integration tests.

See docs/runner_tests_prd.md for requirements.
"""

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from cheat_at_search.search import run_strategy

from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.runners.diff import DiffParams, diff_benchmark
from exps.runners.run import RunParams, run_benchmark
from exps.runners.train import TrainParams, train_strategy
from exps.strategy_config import load_strategy_config, resolve_strategy_class


def _write_fixture_config(tmp_path: Path, fixture_name: str, run_path: Path) -> Path:
    template_path = Path("tests/fixtures/configs") / fixture_name
    content = template_path.read_text(encoding="utf-8")
    content = content.replace("__RUN_PATH__", str(run_path))
    run_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / fixture_name
    config_path.write_text(content, encoding="utf-8")
    return config_path


def _load_rounds(path: Path) -> list[dict]:
    rounds_path = path / "rounds.jsonl"
    payload = rounds_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in payload if line.strip()]


def test_run_benchmark_wands_bm25_all_params():
    params = RunParams(
        strategy_path="configs/bm25.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
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
        dataset="doug_blog",
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
        dataset="doug_blog",
        query="bm25",
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
        dataset="doug_blog",
        query="bm25",
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
        dataset="doug_blog",
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


def test_train_codegen_guarded_wands_ndcg_nonzero(tmp_path: Path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    run_path = tmp_path / "codegen_guarded_wands"
    config_path = _write_fixture_config(
        tmp_path,
        "codegen_guarded_wands_small.yml",
        run_path,
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="wands",
        num_queries=2,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    rounds = _load_rounds(Path(result.artifact_path))
    assert rounds[0]["mean_ndcg"] > 0.0


def test_train_codegen_start_code_rerank_only_wrapper(tmp_path: Path):
    run_path = tmp_path / "codegen_rerank_only"
    config_path = _write_fixture_config(
        tmp_path,
        "codegen_start_code_rerank_only_path.yml",
        run_path,
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=0,
    )
    result = train_strategy(params)

    reranker_path = Path(result.artifact_path) / "reranker.py"
    content = reranker_path.read_text(encoding="utf-8")
    assert Path(result.artifact_path) == run_path
    assert "def reranker(" in content
    assert "def rerank_doug_blog(" in content


def test_train_codegen_path_uses_start_code(tmp_path: Path):
    run_path = tmp_path / "codegen_start_code_marker"
    config_path = _write_fixture_config(
        tmp_path,
        "codegen_start_code_path_marker.yml",
        run_path,
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=0,
    )
    result = train_strategy(params)

    reranker_path = Path(result.artifact_path) / "reranker.py"
    content = reranker_path.read_text(encoding="utf-8")
    assert Path(result.artifact_path) == run_path
    assert "START_CODE_SENTINEL = True" in content


def test_train_codegen_validation_guard_toggle(tmp_path: Path):
    run_path_on = tmp_path / "codegen_validation_on"
    config_path_on = _write_fixture_config(
        tmp_path,
        "codegen_validation_on.yml",
        run_path_on,
    )
    params_on = TrainParams(
        strategy_path=str(config_path_on),
        base_path=None,
        dataset="doug_blog",
        num_queries=2,
        seed=123,
        workers=1,
        device=None,
        rounds=0,
    )
    result_on = train_strategy(params_on)
    metadata_on = json.loads(
        (Path(result_on.artifact_path) / "metadata.json").read_text(encoding="utf-8")
    )
    rounds_on = _load_rounds(Path(result_on.artifact_path))
    assert metadata_on["num_validation_queries"] > 0
    assert rounds_on[0]["validation_query_count"] > 0

    run_path_off = tmp_path / "codegen_validation_off"
    config_path_off = _write_fixture_config(
        tmp_path,
        "codegen_validation_off.yml",
        run_path_off,
    )
    params_off = TrainParams(
        strategy_path=str(config_path_off),
        base_path=None,
        dataset="doug_blog",
        num_queries=2,
        seed=123,
        workers=1,
        device=None,
        rounds=0,
    )
    result_off = train_strategy(params_off)
    metadata_off = json.loads(
        (Path(result_off.artifact_path) / "metadata.json").read_text(encoding="utf-8")
    )
    rounds_off = _load_rounds(Path(result_off.artifact_path))
    assert metadata_off["num_validation_queries"] == 0
    assert rounds_off[0]["validation_query_count"] == 0


def test_run_codegen_raw_tool_list_runner(tmp_path: Path):
    run_path = tmp_path / "codegen_raw_tool_list"
    config_path = _write_fixture_config(
        tmp_path,
        "codegen_raw_tool_list.yml",
        run_path,
    )
    train_params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=0,
    )
    train_strategy(train_params)

    run_params = RunParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(run_params)

    assert result.metric_series is not None
    assert not result.metric_series.empty


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


def test_run_benchmark_agentic_bash_tool(tmp_path, monkeypatch):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")
    if not _docker_available():
        pytest.skip("Docker is required for bash tool integration test.")

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
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        no_cache=True,
    )
    result = run_benchmark(params)

    assert result.metric_series is not None
    assert len(result.metric_series) == 1
