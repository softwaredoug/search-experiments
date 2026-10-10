from __future__ import annotations

import csv
import json

from scripts.plot_ecom_judge_variants import (
    _collect_tool_calls,
    _pareto_front,
    _query_tool_calls,
    _read_pareto_points,
    plot_pareto_results,
)


def _write_trace(
    trace_root,
    *,
    timestamp: str,
    query_slug: str,
    num_tool_calls: int,
    query: str | None = None,
):
    summary_path = (
        trace_root
        / "esci"
        / "agentic_ecom_bm25_fewshot_judge"
        / timestamp
        / query_slug
        / "summary.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"num_tool_calls": num_tool_calls}
    if query is not None:
        summary["query"] = query
    summary_path.write_text(json.dumps(summary), encoding="utf-8")


def test_query_tool_calls_uses_latest_trace_for_each_query_and_skips_empty_runs(
    tmp_path,
):
    trace_root = tmp_path / "agentic"
    _write_trace(
        trace_root,
        timestamp="20260601000000",
        query_slug="blue_chair",
        num_tool_calls=2,
        query="blue chair",
    )
    _write_trace(
        trace_root,
        timestamp="20260601000000",
        query_slug="green_chair",
        num_tool_calls=3,
    )
    _write_trace(
        trace_root,
        timestamp="20260602000000",
        query_slug="blue_chair",
        num_tool_calls=5,
        query="blue chair",
    )
    empty_run = (
        trace_root
        / "esci"
        / "agentic_ecom_bm25_fewshot_judge"
        / "20260603000000"
    )
    empty_run.mkdir(parents=True)

    rows = _query_tool_calls(
        dataset="esci",
        strategy="agentic_ecom_bm25_fewshot_judge",
        trace_root=trace_root,
    )

    calls = {row["query_slug"]: row["num_tool_calls"] for row in rows}
    assert calls == {"blue_chair": 5, "green_chair": 3}
    assert next(row for row in rows if row["query_slug"] == "blue_chair")["query"] == (
        "blue chair"
    )


def test_collect_tool_calls_limits_rows_to_agentic_strategies(tmp_path):
    trace_root = tmp_path / "agentic"
    _write_trace(
        trace_root,
        timestamp="20260601000000",
        query_slug="blue_chair",
        num_tool_calls=4,
    )
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "strategy_name", "metric_name"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "esci",
                "strategy_name": "agentic_ecom_bm25_fewshot_judge",
                "metric_name": "NDCG",
            }
        )
        writer.writerow(
            {"dataset": "esci", "strategy_name": "bm25", "metric_name": "NDCG"}
        )

    rows = _collect_tool_calls(summary_path, trace_root=trace_root)

    assert len(rows) == 1
    assert rows[0]["strategy_name"] == "agentic_ecom_bm25_fewshot_judge"
    assert rows[0]["num_tool_calls"] == 4


def test_pareto_points_use_per_query_trace_calls_over_cached_summary_calls(tmp_path):
    summary_path = tmp_path / "summary.csv"
    tool_calls_path = tmp_path / "tool_calls.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "strategy_name",
                "metric_name",
                "mean_ndcg",
                "tool_calls_mean",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "dataset": "esci",
                    "strategy_name": "bm25",
                    "metric_name": "NDCG",
                    "mean_ndcg": 0.3,
                    "tool_calls_mean": 1,
                },
                {
                    "dataset": "esci",
                    "strategy_name": "agentic_ecom_bm25_fewshot_judge",
                    "metric_name": "NDCG",
                    "mean_ndcg": 0.45,
                    "tool_calls_mean": 0,
                },
            ]
        )
    with tool_calls_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["dataset", "strategy_name", "num_tool_calls"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "dataset": "esci",
                    "strategy_name": "agentic_ecom_bm25_fewshot_judge",
                    "num_tool_calls": 3,
                },
                {
                    "dataset": "esci",
                    "strategy_name": "agentic_ecom_bm25_fewshot_judge",
                    "num_tool_calls": 5,
                },
            ]
        )

    points = _read_pareto_points(summary_path, tool_calls_path)["esci"]

    judge_point = next(
        point
        for point in points
        if point["strategy"] == "agentic_ecom_bm25_fewshot_judge"
    )
    assert judge_point["tool_calls_mean"] == 4
    assert judge_point["mean_ndcg"] == 0.45


def test_pareto_front_keeps_lower_call_higher_ndcg_points():
    points = [
        {"strategy": "a", "tool_calls_mean": 1, "mean_ndcg": 0.3},
        {"strategy": "b", "tool_calls_mean": 2, "mean_ndcg": 0.4},
        {"strategy": "c", "tool_calls_mean": 3, "mean_ndcg": 0.35},
        {"strategy": "d", "tool_calls_mean": 4, "mean_ndcg": 0.5},
    ]

    frontier = _pareto_front(points)

    assert [point["strategy"] for point in frontier] == ["a", "b", "d"]


def test_plot_pareto_results_writes_an_image(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "strategy_name",
                "metric_name",
                "mean_ndcg",
                "tool_calls_mean",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "esci",
                "strategy_name": "bm25",
                "metric_name": "NDCG",
                "mean_ndcg": 0.3,
                "tool_calls_mean": 1,
            }
        )
    output_path = tmp_path / "plots" / "pareto.png"

    plot_pareto_results(summary_path, None, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
