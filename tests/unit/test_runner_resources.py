import csv
from types import SimpleNamespace

import pandas as pd
import pytest

from exps.runner import _report_metric, _write_summary_csv
from exps.runners.run import _resource_summary, _resource_values_by_query


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
            metric_name="NDCG",
            summary={
                "mean_ndcg": 0.4,
                "mean_resource.tool_calls": 3,
            },
        ),
    )

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert "mean_resource.tool_calls" in reader.fieldnames
    assert rows[0]["mean_resource.tool_calls"] == ""
    assert rows[1]["mean_resource.tool_calls"] == "3"
    assert rows[1]["tool_calls_mean"] == ""


def test_summary_output_includes_generic_resource_statistics(capsys):
    _report_metric(
        "NDCG",
        pd.Series([0.3, 0.5]),
        {
            "mean_ndcg": 0.4,
            "median_ndcg": 0.4,
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
    assert "mean_ndcg: 0.4000" in output
    assert "mean_agreement: 0.8000" in output
    assert "# Resources" in output
    assert "mean_tool_calls: 2.0000" in output
    assert "median_tool_calls: 2.0000" in output
    assert "std_tool_calls: 1.0000" in output
    assert "mean_input_tokens: 12.0000" in output
    assert "resource.tool_calls" not in output
    assert "resource.input_tokens" not in output
