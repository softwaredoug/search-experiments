import csv
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from exps.runner import (
    _report_metric,
    _write_full_results_parquet,
    _write_per_query_csv,
    _write_summary_csv,
    main,
)
from exps.runners.run import (
    _mean_max_ndcg_m,
    _resource_summary,
    _resource_values_by_query,
)


def test_resource_summary_rolls_up_once_per_query_and_ignores_missing_values():
    graded = pd.DataFrame(
        {
            "query": ["same query", "same query", "same query", "other", "other"],
            "query_id": [1, 1, 2, 3, 3],
            "rank": [1, 2, 1, 1, 2],
            "resource.tool_calls": [1, 1, 3, None, None],
            "resource.latency": [0.2, 0.2, 0.4, 0.6, 0.6],
        }
    )

    summary = _resource_summary(_resource_values_by_query(graded))

    assert summary == {
        "mean_resource.tool_calls": 2.0,
        "median_resource.tool_calls": 2.0,
        "std_resource.tool_calls": 1.0,
        "mean_resource.latency": pytest.approx(0.4),
        "median_resource.latency": 0.4,
        "std_resource.latency": pytest.approx(0.1632993161855452),
    }


def test_resource_values_reject_different_values_for_one_query():
    graded = pd.DataFrame(
        {
            "query": ["query", "query"],
            "query_id": [1, 1],
            "resource.tool_calls": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="one value per query"):
        _resource_values_by_query(graded)


def test_resource_values_reject_non_numeric_data():
    graded = pd.DataFrame(
        {
            "query": ["query"],
            "query_id": [1],
            "resource.tool_calls": ["unknown"],
        }
    )

    with pytest.raises(ValueError):
        _resource_values_by_query(graded)


def test_mean_max_ndcg_m_averages_once_per_query():
    graded = pd.DataFrame(
        {
            "query": ["first", "first", "second"],
            "query_id": [1, 1, 2],
            "rank": [1, 2, 1],
            "max_ndcg_m": [0.8, 0.8, 0.4],
        }
    )

    assert _mean_max_ndcg_m(graded) == pytest.approx(0.6)


def test_mean_max_ndcg_m_requires_metric_from_cheat_at_search():
    graded = pd.DataFrame({"query": ["query"], "query_id": [1]})

    with pytest.raises(ValueError, match="missing max_ndcg_m"):
        _mean_max_ndcg_m(graded)


def test_resource_summary_ignores_non_resource_columns():
    graded = pd.DataFrame(
        {
            "query": ["query"],
            "query_id": [1],
            "latency": [0.2],
            "resource.tokens": [10],
        }
    )

    assert _resource_summary(_resource_values_by_query(graded)) == {
        "mean_resource.tokens": 10.0,
        "median_resource.tokens": 10.0,
        "std_resource.tokens": 0.0,
    }


def test_summary_csv_adds_resource_columns_to_existing_reports(tmp_path):
    path = tmp_path / "summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "strategy_name", "mean_ndcg", "tool_calls_mean"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "esci",
                "strategy_name": "old_run",
                "mean_ndcg": 0.3,
                "tool_calls_mean": 1,
            }
        )

    _write_summary_csv(
        str(path),
        dataset="esci",
        result=SimpleNamespace(
            strategy_name="agentic",
            strategy_params={},
            metric_name="NDCG_M",
            summary={
                "mean_ndcg_g": 0.35,
                "median_ndcg_g": 0.35,
                "mean_ndcg_m": 0.4,
                "median_ndcg_m": 0.4,
                "mean_max_ndcg_m": 0.9,
                "mean_resource.tool_calls": 3,
            },
        ),
    )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert "mean_resource.tool_calls" in reader.fieldnames
    assert rows[1]["metric_name"] == "NDCG_M"
    assert "mean_ndcg_g" in reader.fieldnames
    assert "median_ndcg_g" in reader.fieldnames
    assert "mean_ndcg_m" in reader.fieldnames
    assert "median_ndcg_m" in reader.fieldnames
    assert "mean_max_ndcg_m" in reader.fieldnames
    assert rows[0]["mean_resource.tool_calls"] == ""
    assert rows[1]["mean_resource.tool_calls"] == "3"
    assert rows[0]["mean_ndcg_g"] == ""
    assert rows[1]["mean_ndcg_g"] == "0.35"
    assert rows[0]["mean_ndcg_m"] == ""
    assert rows[1]["mean_ndcg_m"] == "0.4"
    assert rows[0]["mean_max_ndcg_m"] == ""
    assert rows[1]["mean_max_ndcg_m"] == "0.9"
    assert rows[1]["tool_calls_mean"] == ""


def _per_query_graded_results():
    return pd.DataFrame(
        {
            "query": ["blue chair", "blue chair", "wood table"],
            "query_id": [11, 11, 12],
            "doc_id": [101, 102, 201],
            "rank": [1, 2, 1],
            "ndcg_m": [0.8, 0.8, 0.2],
            "ndcg_g": [0.9, 0.9, 0.4],
            "max_ndcg_m": [0.95, 0.95, 0.6],
            "mrr": [1.0, 1.0, 0.0],
            "agreement": [0.75, 0.75, 0.5],
            "pairwise_agreement": [0.8, 0.8, 0.5],
            "resource.input_tokens": [100, 100, 200],
            "resource.output_tokens": [20, 20, 30],
            "resource.tool_calls": [2, 2, 3],
        }
    )


def test_per_query_csv_collapses_document_rows_and_appends_run_metadata(tmp_path):
    path = tmp_path / "per_query.csv"
    result = SimpleNamespace(
        strategy_name="agentic_gpt6_luna_bm25",
        metric_name="NDCG_M",
        graded=_per_query_graded_results(),
    )

    for seed, run_id in ((43, "run-a"), (44, "run-b")):
        _write_per_query_csv(
            str(path),
            dataset="wands",
            seed=seed,
            run_id=run_id,
            commit_sha="abc123",
            result=result,
        )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert len(rows) == 4
    assert rows[0]["run_id"] == "run-a"
    assert rows[0]["seed"] == "43"
    assert rows[0]["query_id"] == "11"
    assert rows[0]["query"] == "blue chair"
    assert rows[0]["ndcg_m"] == "0.8"
    assert rows[0]["ndcg_g"] == "0.9"
    assert rows[0]["max_ndcg_m"] == "0.95"
    assert rows[0]["mrr"] == "1.0"
    assert rows[0]["resource.tool_calls"] == "2"
    assert rows[1]["query_id"] == "12"
    assert rows[2]["run_id"] == "run-b"
    assert rows[2]["seed"] == "44"


def test_full_results_parquet_writes_document_rows_and_run_metadata(tmp_path):
    path = tmp_path / "full_results.parquet"
    graded = _per_query_graded_results().assign(
        score=[0.9, 0.8, 0.7],
        grade=[4, 2, 1],
        title=["Blue chair A", "Blue chair B", "Wood table"],
        description=["First", "Second", "Third"],
    )
    graded = pd.concat(
        [graded, graded.iloc[[0]].assign(rank=11)], ignore_index=True
    )
    result = SimpleNamespace(
        strategy_name="agentic_gpt6_luna_bm25",
        strategy_params={"model": "gpt-6-luna"},
        metric_name="NDCG_M",
        graded=graded,
    )

    _write_full_results_parquet(
        str(path),
        dataset="wands",
        seed=43,
        run_id="run-a",
        commit_sha="abc123",
        result=result,
    )

    rows = pd.read_parquet(path)
    assert len(rows) == 3
    assert rows["query_id"].tolist() == [11, 11, 12]
    assert rows["rank"].tolist() == [1, 2, 1]
    assert rows["title"].tolist() == ["Blue chair A", "Blue chair B", "Wood table"]
    assert rows["grade"].tolist() == [4, 2, 1]
    assert rows["dataset"].tolist() == ["wands"] * 3
    assert rows["seed"].tolist() == [43] * 3
    assert rows["run_id"].tolist() == ["run-a"] * 3
    assert rows["commit_sha"].tolist() == ["abc123"] * 3
    assert rows["strategy_params"].tolist() == ['{"model": "gpt-6-luna"}'] * 3


def test_run_cli_writes_per_query_csv_with_summary_run_metadata(
    tmp_path, monkeypatch
):
    summary_path = tmp_path / "summary.csv"
    per_query_path = tmp_path / "per_query.csv"
    full_results_path = tmp_path / "full_results.parquet"
    result = SimpleNamespace(
        strategy_name="agentic_gpt6_luna_bm25",
        strategy_params={"model": "gpt-6-luna"},
        metric_name="NDCG_M",
        metric_series=pd.Series([0.8, 0.2]),
        summary={"mean_ndcg_m": 0.5},
        graded=_per_query_graded_results(),
        codegen_artifact_path=None,
        artifact_path=None,
        relevant_examples=None,
    )
    monkeypatch.setattr("exps.runner.run_benchmark", lambda params: result)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--strategy",
            "unused.yml",
            "--dataset",
            "wands",
            "--seed",
            "43",
            "--summary-csv",
            str(summary_path),
            "--per-query-csv",
            str(per_query_path),
            "--full-results-parquet",
            str(full_results_path),
        ],
    )

    main()

    with summary_path.open(newline="", encoding="utf-8") as handle:
        summary_row = next(csv.DictReader(handle))
    with per_query_path.open(newline="", encoding="utf-8") as handle:
        per_query_row = next(csv.DictReader(handle))
    full_results = pd.read_parquet(full_results_path)

    assert summary_row["run_id"] == per_query_row["run_id"]
    assert summary_row["seed"] == per_query_row["seed"] == "43"
    assert per_query_row["ndcg_m"] == "0.8"
    assert full_results["run_id"].unique().tolist() == [summary_row["run_id"]]
    assert full_results["seed"].unique().tolist() == [43]


def test_run_cli_rejects_full_results_for_single_query(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--strategy",
            "unused.yml",
            "--query",
            "blue chair",
            "--full-results-parquet",
            "full_results.parquet",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        main()


def test_summary_output_includes_generic_resource_statistics(capsys):
    _report_metric(
        "NDCG_M",
        pd.Series([0.3, 0.5]),
        {
            "mean_ndcg_g": 0.3,
            "median_ndcg_g": 0.3,
            "mean_ndcg_m": 0.4,
            "median_ndcg_m": 0.4,
            "mean_max_ndcg_m": 0.9,
            "mean_agreement": 0.8,
            "mean_pairwise_agreement": 0.9,
            "mean_resource.tool_calls": 2.0,
            "median_resource.tool_calls": 2.0,
            "std_resource.tool_calls": 1.0,
            "mean_resource.input_tokens": 12.0,
        },
    )

    output = capsys.readouterr().out
    assert "# Summary" in output
    assert "Retrieval Metrics:" in output
    assert "mean_ndcg_g: 0.3000" in output
    assert "median_ndcg_g: 0.3000" in output
    assert "mean_ndcg_m: 0.4000" in output
    assert "median_ndcg_m: 0.4000" in output
    assert "mean_max_ndcg_m (judged ceiling): 0.9000" in output
    assert "mean_agreement: 0.8000" in output
    assert "# Resources" in output
    assert "mean_tool_calls: 2.0000" in output
    assert "median_tool_calls: 2.0000" in output
    assert "std_tool_calls: 1.0000" in output
    assert "mean_input_tokens: 12.0000" in output
    assert "resource.tool_calls" not in output
    assert "resource.input_tokens" not in output
