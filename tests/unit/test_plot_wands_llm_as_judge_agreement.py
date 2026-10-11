from __future__ import annotations

import csv

import pytest

from scripts.plot_wands_llm_as_judge_agreement import (
    _read_agreement_means,
    plot_agreement,
)


def _write_report(path, rows):
    fields = [
        "dataset",
        "strategy_name",
        "metric_name",
        "mean_agreement",
        "mean_pairwise_agreement",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_reads_and_averages_wands_agreement_metrics(tmp_path):
    report_path = tmp_path / "report.csv"
    _write_report(
        report_path,
        [
            {
                "dataset": "wands",
                "strategy_name": "wands_pointwise_judge_gpt5_mini",
                "metric_name": "NDCG",
                "mean_agreement": 0.8,
                "mean_pairwise_agreement": 0.9,
            },
            {
                "dataset": "wands",
                "strategy_name": "wands_pointwise_judge_gpt5_mini",
                "metric_name": "NDCG",
                "mean_agreement": 0.6,
                "mean_pairwise_agreement": 0.7,
            },
            {
                "dataset": "wands",
                "strategy_name": "wands_pointwise_judge_gpt5",
                "metric_name": "NDCG",
                "mean_agreement": 0.7,
                "mean_pairwise_agreement": 0.8,
            },
        ],
    )

    means = _read_agreement_means(report_path)

    assert means["wands_pointwise_judge_gpt5_mini", "mean_agreement"] == pytest.approx(
        0.7
    )
    assert means[
        "wands_pointwise_judge_gpt5_mini", "mean_pairwise_agreement"
    ] == pytest.approx(0.8)
    assert means["wands_pointwise_judge_gpt5", "mean_agreement"] == pytest.approx(
        0.7
    )


def test_rejects_non_wands_reports(tmp_path):
    report_path = tmp_path / "report.csv"
    _write_report(
        report_path,
        [
            {
                "dataset": "esci",
                "strategy_name": "wands_pointwise_judge_gpt5_mini",
                "metric_name": "NDCG",
                "mean_agreement": 0.8,
                "mean_pairwise_agreement": 0.9,
            }
        ],
    )

    with pytest.raises(ValueError, match="only WANDS"):
        _read_agreement_means(report_path)


def test_plot_writes_agreement_chart(tmp_path):
    report_path = tmp_path / "report.csv"
    output_path = tmp_path / "assets" / "agreement.png"
    _write_report(
        report_path,
        [
            {
                "dataset": "wands",
                "strategy_name": strategy,
                "metric_name": "NDCG",
                "mean_agreement": 0.8,
                "mean_pairwise_agreement": 0.9,
            }
            for strategy in (
                "wands_pointwise_judge_gpt5_mini",
                "wands_pointwise_judge_gpt5",
            )
        ],
    )

    plot_agreement(report_path, output_path)

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
