from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer

from exps.runners.run import RunParams, run_benchmark
from exps.runners.train import TrainParams, train_strategy


def _dataset():
    queries = ["amber", "birch", "cedar", "delta", "elm", "fir"]
    corpus = pd.DataFrame(
        {
            "doc_id": list(range(12)),
            "title": [value for query in queries for value in (query, "common")],
            "description": ["common text"] * 12,
        }
    )
    corpus["title_snowball"] = SearchArray.index(corpus["title"], snowball_tokenizer)
    corpus["description_snowball"] = SearchArray.index(
        corpus["description"], snowball_tokenizer
    )
    judgments = pd.DataFrame(
        {
            "query_id": list(range(6)),
            "query": queries,
            "doc_id": list(range(0, 12, 2)),
            "grade": [2] * 6,
        }
    )
    return SimpleNamespace(corpus=corpus, judgments=judgments)


def _write_config(path, *, run_path=None):
    strategy_path = "" if run_path is None else f"  path: {run_path}\n"
    path.write_text(
        f"""
strategy:
  name: random_search_fixture
  type: random_search
{strategy_path}  params:
    retrieval_engine:
      base: bm25
      params:
        k1: "{{k1}}"
        b: 0.75
        title_boost: "{{title_boost}}"
        description_boost: 1.0
    train:
      rounds: 4
      search_seed: 21
      search_space:
        k1:
          distribution: choice
          values: [0.5, 1.5]
        title_boost:
          distribution: choice
          values: [1.0, 10.0]
      objective: ndcg
      eval:
        train_fraction: 0.5
        validation_fraction: 0.25
        seed: 23
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_random_search_training_continuation_and_runtime(
    tmp_path, monkeypatch
):
    from exps import paths
    from exps.runners import run as run_module
    from exps.runners import train as train_module
    from exps import train_runs

    dataset = _dataset()
    monkeypatch.setattr(paths, "SEARCH_EXPERIMENTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(train_runs, "SEARCH_EXPERIMENTS_ROOT", tmp_path / "artifacts")
    monkeypatch.setattr(train_module, "get_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(run_module, "get_dataset", lambda *args, **kwargs: dataset)

    initial_config = _write_config(tmp_path / "random_search.yml")
    result = train_strategy(
        TrainParams(
            strategy_path=str(initial_config),
            dataset="wands",
            num_queries=5,
            rounds=2,
            seed=37,
            workers=1,
            device="cpu",
        )
    )
    assert result.artifact_path
    assert result.metadata["rounds"] == 2
    assert result.metadata["base_query_count"] == 5
    assert result.metadata["train_query_count"] > 0
    assert result.metadata["validation_query_count"] > 0
    assert result.metadata["test_query_count"] > 0
    assert (tmp_path / "artifacts" / "random_search" / "wands" / "random_search_fixture").exists()

    continued_config = _write_config(
        tmp_path / "random_search_continue.yml", run_path=result.artifact_path
    )
    continued = train_strategy(
        TrainParams(
            strategy_path=str(continued_config),
            dataset="wands",
            rounds=1,
            seed=37,
            workers=1,
            device="cpu",
        )
    )
    assert continued.artifact_path == result.artifact_path
    assert continued.metadata["rounds"] == 3
    assert continued.metadata["rounds_added"] == 1
    assert len((tmp_path / continued.artifact_path / "trials.jsonl").read_text().splitlines()) == 3
    trial_records = [
        json.loads(line)
        for line in (tmp_path / continued.artifact_path / "trials.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    completed = [trial for trial in trial_records if trial["status"] == "completed"]
    assert continued.metadata["winning_round"] == max(
        completed, key=lambda trial: (trial["validation_ndcg"], -trial["round"])
    )["round"]

    runtime_config = _write_config(tmp_path / "random_search_run.yml")
    run_result = run_benchmark(
        RunParams(
            strategy_path=str(runtime_config),
            dataset="wands",
            workers=1,
            seed=37,
            device="cpu",
            no_cache=True,
        )
    )
    assert run_result.artifact_path == continued.artifact_path
    assert run_result.metric_series is not None
    assert len(run_result.metric_series) == 6
