import csv

import pytest

from scripts.plot_ecom_comp_variants import STRATEGIES, _read_mean_ndcg


def test_read_mean_ndcg_averages_runs_for_each_dataset_and_strategy(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "strategy_name", "metric_name", "mean_ndcg"],
        )
        writer.writeheader()
        for dataset in ("esci", "wands"):
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
                "dataset": "esci",
                "strategy_name": "bm25",
                "metric_name": "NDCG",
                "mean_ndcg": 0.4,
            }
        )
        writer.writerow(
            {
                "dataset": "esci",
                "strategy_name": "bm25",
                "metric_name": "MRR",
                "mean_ndcg": 0.9,
            }
        )

    means = _read_mean_ndcg(summary_path)

    assert means[("esci", "bm25")] == pytest.approx(0.3)
    assert means[("wands", "bm25")] == 0.2
    assert means[("esci", "agentic_ecom_composite_rrf_jev_gpt5_mini")] == 0.2
