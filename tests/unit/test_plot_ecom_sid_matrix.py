import csv
import sys

import pytest

from scripts.plot_ecom_sid_matrix import (
    GPT5_STRATEGIES,
    PARETO_STRATEGIES,
    STRATEGIES,
    _pareto_front,
    _read_mean_ndcg,
    _read_mean_output_tokens,
    _read_mean_token_cost_usd,
    main,
)


def test_matrix_includes_all_model_variants():
    strategy_labels = dict(STRATEGIES)

    assert strategy_labels["agentic_bm25_ecommerce"] == "GPT-5 · BM25"
    assert strategy_labels["agentic_bm25_e5_ecommerce_gpt5"] == "GPT-5 · BM25 + E5"
    assert strategy_labels["agentic_bm25_ecommerce_gpt5_mini"] == "GPT-5-mini · BM25"
    assert (
        strategy_labels["agentic_bm25_e5_ecommerce_gpt5_mini"]
        == "GPT-5-mini · BM25 + E5"
    )
    assert strategy_labels["agentic_bm25_ecommerce_gpt6_luna"] == "GPT-6 Luna · BM25"
    assert (
        strategy_labels["agentic_bm25_e5_ecommerce_gpt6_luna"]
        == "GPT-6 Luna · BM25 + E5"
    )


def test_pareto_peers_exclude_full_gpt5_but_keep_gpt5_mini_and_sid():
    pareto_strategy_names = {strategy for strategy, _ in PARETO_STRATEGIES}

    assert GPT5_STRATEGIES.isdisjoint(pareto_strategy_names)
    assert "agentic_bm25_ecommerce_gpt5_mini" in pareto_strategy_names
    assert "agentic_bm25_e5_ecommerce_gpt5_mini" in pareto_strategy_names
    assert "agentic_bm25_ecommerce_gpt6_luna" in pareto_strategy_names
    assert "agentic_bm25_e5_ecommerce_gpt6_luna" in pareto_strategy_names
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


def test_read_mean_output_tokens_ignores_missing_resource_values(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "strategy_name",
                "metric_name",
                "mean_resource.output_tokens",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "bm25",
                "metric_name": "NDCG",
                "mean_resource.output_tokens": "",
            }
        )
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "agentic_ecom_sid_bm25_medium",
                "metric_name": "NDCG",
                "mean_resource.output_tokens": "256",
            }
        )

    assert _read_mean_output_tokens(summary_path) == {
        ("wands", "agentic_ecom_sid_bm25_medium"): 256.0
    }


def test_read_mean_token_cost_uses_model_rates_and_cached_input_when_available(
    tmp_path,
):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "strategy_name",
                "metric_name",
                "mean_resource.input_tokens",
                "mean_resource.cached_input_tokens",
                "mean_resource.output_tokens",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "agentic_bm25_ecommerce_gpt5_mini",
                "metric_name": "NDCG",
                "mean_resource.input_tokens": 2_000_000,
                "mean_resource.cached_input_tokens": 1_000_000,
                "mean_resource.output_tokens": 1_000_000,
            }
        )
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "agentic_bm25_ecommerce_gpt6_luna",
                "metric_name": "NDCG",
                "mean_resource.input_tokens": 2_000_000,
                "mean_resource.cached_input_tokens": 1_000_000,
                "mean_resource.output_tokens": 1_000_000,
            }
        )
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "agentic_ecom_sid_bm25_medium",
                "metric_name": "NDCG",
                "mean_resource.input_tokens": 2_000_000,
                "mean_resource.cached_input_tokens": 1_000_000,
                "mean_resource.output_tokens": 1_000_000,
            }
        )

    costs = _read_mean_token_cost_usd(summary_path)

    assert costs[("wands", "agentic_bm25_ecommerce_gpt5_mini")] == pytest.approx(
        2.275
    )
    assert costs[("wands", "agentic_ecom_sid_bm25_medium")] == pytest.approx(1.275)
    assert costs[("wands", "agentic_bm25_ecommerce_gpt6_luna")] == pytest.approx(
        0.61
    )


def test_read_mean_token_cost_assumes_uncached_input_when_cache_usage_is_missing(
    tmp_path,
):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "strategy_name",
                "metric_name",
                "mean_resource.input_tokens",
                "mean_resource.output_tokens",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "wands",
                "strategy_name": "agentic_ecom_sid_bm25_medium",
                "metric_name": "NDCG",
                "mean_resource.input_tokens": 1_000_000,
                "mean_resource.output_tokens": 1_000_000,
            }
        )

    costs = _read_mean_token_cost_usd(summary_path)

    assert costs[("wands", "agentic_ecom_sid_bm25_medium")] == pytest.approx(1.25)


def test_pareto_front_uses_higher_quality_and_lower_output_tokens():
    points = [
        {"strategy": "a", "mean_ndcg": 0.5, "mean_resource.output_tokens": 100.0},
        {"strategy": "b", "mean_ndcg": 0.6, "mean_resource.output_tokens": 300.0},
        {"strategy": "c", "mean_ndcg": 0.7, "mean_resource.output_tokens": 200.0},
        {"strategy": "d", "mean_ndcg": 0.4, "mean_resource.output_tokens": 400.0},
        {"strategy": "e", "mean_ndcg": 0.5, "mean_resource.output_tokens": 200.0},
    ]

    assert [point["strategy"] for point in _pareto_front(points)] == ["a", "c"]


def test_pareto_front_can_use_estimated_token_cost():
    points = [
        {"strategy": "a", "mean_ndcg": 0.5, "mean_token_cost_usd": 0.003},
        {"strategy": "b", "mean_ndcg": 0.6, "mean_token_cost_usd": 0.005},
        {"strategy": "c", "mean_ndcg": 0.7, "mean_token_cost_usd": 0.004},
    ]

    assert [
        point["strategy"]
        for point in _pareto_front(points, "mean_token_cost_usd")
    ] == ["a", "c"]


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
                "mean_resource.input_tokens",
                "mean_resource.output_tokens",
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
                        "mean_resource.input_tokens": 100,
                        "mean_resource.output_tokens": 200,
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
        tmp_path / "plots" / "sid_matrix_cost_pareto_wands.png",
        tmp_path / "plots" / "sid_matrix_cost_pareto_esci.png",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in output_paths)


def test_main_accepts_separate_pareto_input(tmp_path, monkeypatch):
    retrieval_input = tmp_path / "full.csv"
    pareto_input = tmp_path / "spotcheck.csv"
    output_prefix = tmp_path / "plots" / "sid_matrix"
    seen = []
    monkeypatch.setattr(
        "scripts.plot_ecom_sid_matrix.plot_results",
        lambda path, _output, dataset: seen.append(("retrieval", path, dataset)),
    )
    monkeypatch.setattr(
        "scripts.plot_ecom_sid_matrix.plot_pareto_results",
        lambda path, _output, dataset: seen.append(("pareto", path, dataset)),
    )
    monkeypatch.setattr(
        "scripts.plot_ecom_sid_matrix.plot_cost_pareto_results",
        lambda path, _output, dataset: seen.append(("cost_pareto", path, dataset)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_ecom_sid_matrix.py",
            "--input",
            str(retrieval_input),
            "--pareto-input",
            str(pareto_input),
            "--output-prefix",
            str(output_prefix),
        ],
    )

    main()

    assert seen == [
        ("retrieval", retrieval_input, "wands"),
        ("pareto", pareto_input, "wands"),
        ("cost_pareto", pareto_input, "wands"),
        ("retrieval", retrieval_input, "esci"),
        ("pareto", pareto_input, "esci"),
        ("cost_pareto", pareto_input, "esci"),
    ]
