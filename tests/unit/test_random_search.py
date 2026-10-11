import csv
import random

import pytest

from exps.random_search.train import materialize_engine_config, split_queries
from exps.runners.train import TrainResult, _write_summary_csv


def test_materialize_engine_config_samples_placeholders_as_numbers():
    engine = {
        "base": "bm25",
        "params": {
            "k1": "{k1}",
            "b": "{b}",
            "title_boost": 9.3,
            "description_boost": 4.1,
        },
    }
    search_space = {
        "k1": {"distribution": "choice", "values": [0.5, 1.5]},
        "b": {"distribution": "uniform", "min": 0.25, "max": 0.75},
    }

    sampled = materialize_engine_config(engine, search_space, random.Random(17))
    repeated = materialize_engine_config(engine, search_space, random.Random(17))

    assert sampled == repeated
    assert sampled["base"] == "bm25"
    assert set(sampled["params"]) == {"k1", "b", "title_boost", "description_boost"}
    assert sampled["params"]["k1"] in {0.5, 1.5}
    assert 0.25 <= sampled["params"]["b"] <= 0.75
    assert sampled["params"]["title_boost"] == 9.3


@pytest.mark.parametrize(
    ("engine", "search_space", "message"),
    [
        (
            {"base": "bm25", "params": {"k1": "{k1}", "b": 0.75}},
            {},
            "missing search-space",
        ),
        (
            {"base": "bm25", "params": {"k1": 1.2, "b": 0.75}},
            {"k1": {"distribution": "choice", "values": [1.2]}},
            "unused search-space",
        ),
        (
            {"base": "bm25", "params": {"k1": "{k1}", "b": 0.75}},
            {"k1": {"distribution": "choice", "values": [-1.0]}},
            "greater than zero",
        ),
        (
            {"base": "embedding", "params": {"k1": "{k1}", "b": 0.75}},
            {"k1": {"distribution": "choice", "values": [1.2]}},
            "Only retrieval_engine.base 'bm25'",
        ),
    ],
)
def test_materialize_engine_config_rejects_invalid_configuration(
    engine, search_space, message
):
    with pytest.raises(ValueError, match=message):
        materialize_engine_config(engine, search_space, random.Random(1))


def test_split_queries_is_reproducible_and_disjoint():
    queries = [f"query-{index}" for index in range(20)]
    first = split_queries(
        queries, train_fraction=0.6, validation_fraction=0.2, seed=31
    )
    second = split_queries(
        queries, train_fraction=0.6, validation_fraction=0.2, seed=31
    )

    assert first == second
    assert all(first)
    assert [len(split) for split in first] == [12, 4, 4]
    assert len(first[0]) + len(first[1]) + len(first[2]) == len(queries)
    assert not set(first[0]) & set(first[1])
    assert not set(first[0]) & set(first[2])
    assert not set(first[1]) & set(first[2])


def test_split_queries_requires_three_query_holdouts():
    with pytest.raises(ValueError, match="at least three"):
        split_queries(
            ["one", "two"], train_fraction=0.7, validation_fraction=0.15, seed=1
        )


def test_train_summary_csv_uses_generic_artifact_column(tmp_path):
    summary_path = tmp_path / "summary.csv"
    result = TrainResult(
        strategy_name="bm25_fixture",
        strategy_params={"k1": 1.2},
        artifact_path="/tmp/train-artifact",
        metadata={"rounds": 4},
    )

    _write_summary_csv(str(summary_path), dataset="wands", result=result)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        row = next(reader)
    assert "artifact_path" in reader.fieldnames
    assert "codegen_artifact_path" not in reader.fieldnames
    assert row["artifact_path"] == result.artifact_path


def test_train_summary_csv_appends_to_legacy_codegen_schema(tmp_path):
    summary_path = tmp_path / "legacy.csv"
    summary_path.write_text(
        "dataset,commit_sha,strategy_name,strategy_params,codegen_artifact_path,train_metadata\n",
        encoding="utf-8",
    )
    result = TrainResult(
        strategy_name="random_search_fixture",
        strategy_params={},
        artifact_path="/tmp/random-search-artifact",
        metadata={},
    )

    _write_summary_csv(str(summary_path), dataset="wands", result=result)

    with summary_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["codegen_artifact_path"] == result.artifact_path
