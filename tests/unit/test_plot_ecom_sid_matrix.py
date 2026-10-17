import csv

import pytest

from scripts.plot_ecom_sid_matrix import STRATEGIES, _read_mean_ndcg, plot_results


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
    assert means[("wands", "agentic_ecom_sid_bm25_e5_base_v2")] == 0.2


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


def test_plot_results_writes_an_image(tmp_path):
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
                        "mean_ndcg": 0.3,
                    }
                )
    output_path = tmp_path / "plots" / "sid_matrix.png"

    plot_results(summary_path, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
