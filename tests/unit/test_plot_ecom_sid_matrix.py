import csv
import sys

import pytest

from scripts.plot_ecom_sid_matrix import (
    GPT5_STRATEGIES,
    PARETO_STRATEGIES,
    STRATEGIES,
    _pareto_front,
    _read_mean_ndcg,
    _read_mean_tool_calls,
    main,
)


def test_matrix_includes_full_gpt5_and_gpt5_mini_tool_variants():
    strategy_labels = dict(STRATEGIES)

    assert strategy_labels["agentic_bm25_ecommerce"] == "GPT-5 · BM25"
    assert strategy_labels["agentic_bm25_e5_ecommerce_gpt5"] == "GPT-5 · BM25 + E5"
    assert strategy_labels["agentic_bm25_ecommerce_gpt5_mini"] == "GPT-5-mini · BM25"
    assert (
        strategy_labels["agentic_bm25_e5_ecommerce_gpt5_mini"]
        == "GPT-5-mini · BM25 + E5"
    )


def test_pareto_peers_exclude_full_gpt5_but_keep_gpt5_mini_and_sid():
    pareto_strategy_names = {strategy for strategy, _ in PARETO_STRATEGIES}

    assert GPT5_STRATEGIES.isdisjoint(pareto_strategy_names)
    assert "agentic_bm25_ecommerce_gpt5_mini" in pareto_strategy_names
    assert "agentic_bm25_e5_ecommerce_gpt5_mini" in pareto_strategy_names
    assert "agentic_ecom_sid_bm25_low" in pareto_strategy_names


def test_read_mean_ndcg_averages_runs_and_ignores_other_metrics(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "strategy_name", "metric_name", "mean_ndcg"],
        )
        writer.writeheader()
        for dataset in ("wands", "esci"):
            for strategy, _ in STRATEGIES:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "strategy_name": strategy,
                        "metric_name": "NDCG",
                        "mean_ndcg": 0.2,
                    }
                )
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "bm25",
                "metric_name": "NDCG",
                "mean_ndcg": 0.4,
            }
        )
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "bm25",
                "metric_name": "MRR",
                "mean_ndcg": 0.9,
            }
        )

    means = _read_mean_ndcg(summary_path)

    assert means[("wands", "bm25")] == pytest.approx(0.3)
    assert means[("esci", "bm25")] == 0.2
    assert means[("wands", "agentic_ecom_sid_bm25_e5_base_v2_medium")] == 0.2


def test_read_mean_ndcg_reports_missing_strategy_results(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "strategy_name", "metric_name", "mean_ndcg"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "bm25",
                "metric_name": "NDCG",
                "mean_ndcg": 0.3,
            }
        )

    with pytest.raises(ValueError, match="No mean NDCG results found"):
        _read_mean_ndcg(summary_path)


def test_read_mean_tool_calls_ignores_missing_resource_values(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "strategy_name",
                "metric_name",
                "mean_resource.tool_calls",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "bm25",
                "metric_name": "NDCG",
                "mean_resource.tool_calls": "",
            }
        )
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "agentic_ecom_sid_bm25_medium",
                "metric_name": "NDCG",
                "mean_resource.tool_calls": "2",
            }
        )

    assert _read_mean_tool_calls(summary_path) == {
        ("wands", "agentic_ecom_sid_bm25_medium"): 2.0
    }


def test_pareto_front_uses_higher_quality_and_lower_resource_cost():
    points = [
        {"strategy": "a", "mean_ndcg": 0.5, "mean_resource.tool_calls": 1.0},
        {"strategy": "b", "mean_ndcg": 0.6, "mean_resource.tool_calls": 3.0},
        {"strategy": "c", "mean_ndcg": 0.7, "mean_resource.tool_calls": 2.0},
        {"strategy": "d", "mean_ndcg": 0.4, "mean_resource.tool_calls": 4.0},
        {"strategy": "e", "mean_ndcg": 0.5, "mean_resource.tool_calls": 2.0},
    ]

    assert [point["strategy"] for point in _pareto_front(points)] == ["a", "c"]


def test_main_writes_separate_retrieval_and_pareto_images_per_dataset(
    tmp_path, monkeypatch
):
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
        for dataset in ("wands", "esci"):
            for strategy, _ in STRATEGIES:
                writer.writerow(
                    {
                        "dataset": dataset,
                        "strategy_name": strategy,
                        "metric_name": "NDCG",
                        "mean_ndcg": 0.3,
                        "mean_resource.tool_calls": 2,
                    }
                )
    output_prefix = tmp_path / "plots" / "sid_matrix"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ecom_sid_matrix.py",
            "--input",
            str(summary_path),
            "--output-prefix",
            str(output_prefix),
        ],
    )

    main()

    output_paths = [
        tmp_path / "plots" / "sid_matrix_wands.png",
        tmp_path / "plots" / "sid_matrix_esci.png",
        tmp_path / "plots" / "sid_matrix_pareto_wands.png",
        tmp_path / "plots" / "sid_matrix_pareto_esci.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in output_paths)
