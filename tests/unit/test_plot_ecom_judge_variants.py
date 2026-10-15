from __future__ import annotations

import csv
from scripts.plot_ecom_judge_variants import (
    _pareto_front,
    _read_pareto_points,
    plot_pareto_results,
)


def test_pareto_points_use_only_resource_summary(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "strategy_name",
                "metric_name",
                "mean_ndcg",
                "mean_resource.tool_calls",
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
                    "mean_resource.tool_calls": "",
                    "tool_calls_mean": 1,
                },
                {
                    "dataset": "esci",
                    "strategy_name": "agentic_ecom_bm25_fewshot_judge",
                    "metric_name": "NDCG",
                    "mean_ndcg": 0.45,
                    "mean_resource.tool_calls": 4,
                    "tool_calls_mean": 0,
                },
            ]
        )

    points = _read_pareto_points(summary_path)["esci"]

    judge_point = next(
        point
        for point in points
        if point["strategy"] == "agentic_ecom_bm25_fewshot_judge"
    )
    assert judge_point["mean_resource.tool_calls"] == 4
    assert judge_point["mean_ndcg"] == 0.45
    assert {point["strategy"] for point in points} == {
        "agentic_ecom_bm25_fewshot_judge",
    }


def test_pareto_front_keeps_lower_call_higher_ndcg_points():
    points = [
        {"strategy": "a", "mean_resource.tool_calls": 1, "mean_ndcg": 0.3},
        {"strategy": "b", "mean_resource.tool_calls": 2, "mean_ndcg": 0.4},
        {"strategy": "c", "mean_resource.tool_calls": 3, "mean_ndcg": 0.35},
        {"strategy": "d", "mean_resource.tool_calls": 4, "mean_ndcg": 0.5},
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
                "mean_resource.tool_calls",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "esci",
                "strategy_name": "agentic_ecom_bm25_fewshot_judge",
                "metric_name": "NDCG",
                "mean_ndcg": 0.3,
                "mean_resource.tool_calls": 2,
            }
        )
    output_path = tmp_path / "plots" / "pareto.png"

    plot_pareto_results(summary_path, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
