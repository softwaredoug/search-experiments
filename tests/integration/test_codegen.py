"""Codegen integration tests.

See docs/runner_tests_prd.md for requirements.
"""

import os
import shutil
from pathlib import Path

import pytest

from exps.runners.run import RunParams, run_benchmark
from exps.runners.train import TrainParams, train_strategy


def test_run_benchmark_codegen_guarded():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    params = RunParams(
        strategy_path="configs/codegen_guarded.yml",
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


def test_train_codegen_guarded(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    config_path = tmp_path / "codegen_guarded_train.yml"
    config_path.write_text(
        """
strategy:
  name: codegen_guarded_train_fixture
  type: codegen
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
        - minilm
      edit:
        guards:
          - validation
          - length:
              max_lines: 5
              max_cols: 120
      eval:
        train_fraction: 0.2
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.

    run:
      top_k: 5
""".lstrip(),
        encoding="utf-8",
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path


def test_train_codegen_with_get_corpus():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")
    params = TrainParams(
        strategy_path="configs/codegen_get_corpus.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path


def test_train_codegen_raw_only():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")
    params = TrainParams(
        strategy_path="configs/codegen_raw_only.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path


def test_train_codegen_start_code():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")
    params = TrainParams(
        strategy_path="configs/codegen_start_code.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path


def test_train_codegen_start_code_mismatch():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")
    params = TrainParams(
        strategy_path="configs/codegen_start_code_mismatch.yml",
        base_path="tests/fixtures",
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    with pytest.raises(ValueError, match="start_code does not match configured tools"):
        train_strategy(params)


def test_train_codegen_start_code_dedent(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")
    config_path = tmp_path / "codegen_start_code_dedent.yml"
    config_path.write_text(
        """
strategy:
  name: codegen_start_code_dedent_fixture
  type: codegen
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - bm25
      start_code: |
        import numpy as np

        def rerank_doug_blog(query, bm25, **kwargs):
            docs = bm25(query, top_k=5)
            return [doc["id"] for doc in docs]
      edit:
        guards:
          - validation
      eval:
        train_fraction: 0.2
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
""".lstrip(),
        encoding="utf-8",
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path


def test_train_codegen_path_continuation_fixture(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    source_path = Path("tests/fixtures/past_runs/20260502_025238")
    continue_from = tmp_path / "continued_run"
    shutil.copytree(source_path, continue_from)
    config_path = tmp_path / "codegen_continue_path.yml"
    config_path.write_text(
        f"""
strategy:
  name: codegen_continue_path_fixture
  type: codegen
  path: {continue_from}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
      edit:
        guards:
          - validation
      eval:
        train_fraction: 0.2
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
""".lstrip(),
        encoding="utf-8",
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path
    assert result.metadata["continued_from"] == str(Path(continue_from).expanduser())
    assert result.metadata["previous_rounds"] > 0
    assert result.metadata["rounds"] == result.metadata["previous_rounds"] + 1
    round_name = f"reranker_round_{result.metadata['rounds']}.py"
    round_path = Path(result.artifact_path) / round_name
    assert round_path.exists()


def test_train_codegen_path_missing_creates_run(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    missing_path = tmp_path / "nope"
    config_path = tmp_path / "codegen_missing_path.yml"
    config_path.write_text(
        f"""
strategy:
  name: codegen_missing_path_fixture
  type: codegen
  path: {missing_path}
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - fielded_bm25
      edit:
        guards:
          - validation
      eval:
        train_fraction: 0.2
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
""".lstrip(),
        encoding="utf-8",
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path == str(missing_path)
    assert Path(result.artifact_path).exists()


def test_run_codegen_without_trained_run(monkeypatch, tmp_path):
    config_path = tmp_path / "codegen_no_run.yml"
    config_path.write_text(
        """
strategy:
  name: codegen_no_run_fixture
  type: codegen
  params:
    train:
      search_tools:
        - bm25
    run: {}
""".lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setattr("exps.codegen.strategy.find_latest_codegen_run", lambda *_: None)
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
    with pytest.raises(ValueError, match="No trained codegen run found"):
        run_benchmark(params)


def test_train_codegen_raw_tool_list(tmp_path):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for codegen tests.")

    config_path = tmp_path / "codegen_raw_list.yml"
    config_path.write_text(
        """
strategy:
  name: codegen_raw_list_fixture
  type: codegen
  params:
    train:
      model: gpt-5-mini
      reasoning: low
      refresh_every: 1
      search_tools:
        - raw:
            - get_corpus
      edit:
        guards:
          - validation
      eval:
        train_fraction: 0.2
        seed: 123
        eval_margin: 0.0
      system_prompt: |
        Improve the reranker.
    run:
      top_k: 5
""".lstrip(),
        encoding="utf-8",
    )
    params = TrainParams(
        strategy_path=str(config_path),
        base_path=None,
        dataset="doug_blog",
        num_queries=1,
        seed=123,
        workers=1,
        device=None,
        rounds=1,
    )
    result = train_strategy(params)

    assert result.artifact_path
