import csv
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from exps.reporting import write_benchmark_reports
from exps.runner import _report_metric, main
from exps.runners.run import (
    _mean_max_ndcg_m,
    _mark_judged_results,
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


def test_mark_judged_results_distinguishes_unjudged_from_grade_zero():
    graded = pd.DataFrame(
        {
            "query_id": [1, 1, 2],
            "query": ["chair", "chair", "table"],
            "doc_id": [10, 11, 20],
            "grade": [0.0, 0.0, 1.0],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": [1, 2],
            "query": ["chair", "table"],
            "doc_id": [10, 20],
            "grade": [0.0, 1.0],
        }
    )

    marked = _mark_judged_results(graded, judgments)

    assert marked["judged"].tolist() == [True, False, True]
    assert marked["grade"].tolist() == [0.0, 0.0, 1.0]


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


def _per_query_graded_results():
    return pd.DataFrame(
        {
            "query": ["blue chair", "blue chair", "wood table"],
            "query_id": [11, 11, 12],
            "doc_id": [101, 102, 201],
            "rank": [1, 2, 1],
            "grade": [1, 0, 1],
            "judged": [True, False, True],
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


def _report_result():
    graded = _per_query_graded_results().assign(
        score=[0.9, 0.8, 0.7],
        grade=[4, 0, 0],
        judged=[True, False, True],
        title=["Blue chair A", "Blue chair B", "Wood table"],
        description=["First", "Second", "Third"],
    )
    graded["doc_id"] = pd.Series([101, "ESCI-102", 201], dtype=object)
    graded = pd.concat(
        [graded, graded.iloc[[0]].assign(rank=11)], ignore_index=True
    )
    return SimpleNamespace(
        strategy_name="agentic_gpt6_luna_bm25",
        strategy_params={"model": "gpt-6-luna"},
        metric_name="NDCG_M",
        summary={"mean_ndcg_m": 0.5, "mean_resource.tool_calls": 2.5},
        graded=graded,
    )


def test_benchmark_reports_are_written_under_unique_artifact_run_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("exps.reporting._current_commit_sha", lambda: "abc123")
    artifact_path = tmp_path / "strategy-artifact"
    summary_csv = tmp_path / "results.csv"
    summary_csv.write_text(
        "dataset,strategy_name,mean_ndcg\nwands,previous_run,0.4\n",
        encoding="utf-8",
    )
    result = _report_result()

    first = write_benchmark_reports(
        result,
        dataset="wands",
        seed=43,
        artifact_path=artifact_path,
        summary_csv=summary_csv,
    )
    second = write_benchmark_reports(
        result,
        dataset="wands",
        seed=44,
        artifact_path=artifact_path,
        summary_csv=summary_csv,
    )

    assert first.reports_path == artifact_path / "reports" / first.run_id
    assert second.reports_path == artifact_path / "reports" / second.run_id
    assert first.reports_path != second.reports_path
    for reports_path in (first.reports_path, second.reports_path):
        assert (reports_path / "per_query.csv").exists()
        assert (reports_path / "full_results.parquet").exists()

    with summary_csv.open(newline="", encoding="utf-8") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 3
    assert summary_rows[0]["reports_path"] == ""
    assert summary_rows[1]["reports_path"] == str(first.reports_path)
    assert summary_rows[2]["reports_path"] == str(second.reports_path)
    assert summary_rows[1]["mean_resource.tool_calls"] == "2.5"

    with (first.reports_path / "per_query.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        per_query_rows = list(csv.DictReader(handle))
    assert len(per_query_rows) == 2
    assert per_query_rows[0]["run_id"] == first.run_id
    assert per_query_rows[0]["reports_path"] == str(first.reports_path)
    assert per_query_rows[0]["query_id"] == "11"
    assert per_query_rows[0]["ndcg_m"] == "0.8"

    full_results = pd.read_parquet(first.reports_path / "full_results.parquet")
    assert len(full_results) == 3
    assert full_results["query_id"].tolist() == [11, 11, 12]
    assert full_results["rank"].tolist() == [1, 2, 1]
    assert full_results["doc_id"].tolist() == ["101", "ESCI-102", "201"]
    assert full_results["judged"].tolist() == [True, False, True]
    assert full_results["grade"].iloc[0] == 4
    assert pd.isna(full_results["grade"].iloc[1])
    assert full_results["grade"].iloc[2] == 0
    assert full_results["run_id"].unique().tolist() == [first.run_id]


def test_run_cli_forwards_summary_csv_and_cache_salt_to_backend(monkeypatch, capsys):
    captured = {}
    result = SimpleNamespace(
        metric_name="NDCG_M",
        metric_series=pd.Series([0.5]),
        summary={"mean_ndcg_m": 0.5},
        codegen_artifact_path=None,
        artifact_path=None,
        reports_path="/tmp/reports/run-1",
    )

    def fake_run_benchmark(params):
        captured["params"] = params
        return result

    monkeypatch.setattr("exps.runner.run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--strategy",
            "unused.yml",
            "--summary-csv",
            "results.csv",
            "--cache-salt",
            "round-2",
        ],
    )

    main()

    assert captured["params"].summary_csv == "results.csv"
    assert captured["params"].cache_salt == "round-2"
    assert "Reports: /tmp/reports/run-1" in capsys.readouterr().out


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
