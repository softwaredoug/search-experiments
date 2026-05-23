"""Codegen runner integration tests.

See docs/runner_tests_prd.md for requirements.
"""

import json
import os
from pathlib import Path

import pytest

from exps.runners.run import RunParams, run_benchmark
from exps.runners.train import TrainParams, train_strategy


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
