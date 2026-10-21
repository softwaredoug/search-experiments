import csv
import sys

from scripts.plot_beir_matrix import DATASETS, STRATEGIES, main


def test_matrix_includes_requested_models_and_sid_reasoning_levels():
    labels = [label for _, label in STRATEGIES]

    assert "BM25" in labels
    assert "GPT-5-mini · BM25" in labels
    assert "GPT-5-mini · BM25 + E5" in labels
    assert "GPT-6 Luna · BM25" in labels
    assert "GPT-6 Luna · BM25 + E5" in labels
    assert not any(label.startswith("GPT-5 · ") for label in labels)
    assert {"SID-2 · BM25 · low", "SID-2 · BM25 · medium", "SID-2 · BM25 · high"}.issubset(
        labels
    )
    assert {
        "SID-2 · BM25 + E5 · low",
        "SID-2 · BM25 + E5 · medium",
        "SID-2 · BM25 + E5 · high",
    }.issubset(labels)


def test_main_writes_quality_and_pareto_plots(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.csv"
    fieldnames = [
        "dataset",
        "strategy_name",
        "metric_name",
        "mean_ndcg",
        "mean_resource.input_tokens",
        "mean_resource.cached_input_tokens",
        "mean_resource.output_tokens",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in DATASETS:
            for index, (strategy, _) in enumerate(STRATEGIES):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "strategy_name": strategy,
                        "metric_name": "NDCG",
                        "mean_ndcg": 0.2 + index / 20,
                        "mean_resource.input_tokens": 1000,
                        "mean_resource.cached_input_tokens": 0,
                        "mean_resource.output_tokens": 100 + index * 10,
                    }
                )

    output_prefix = tmp_path / "beir_matrix"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_beir_matrix.py",
            "--input",
            str(summary_path),
            "--output-prefix",
            str(output_prefix),
        ],
    )

    main()

    for dataset in DATASETS:
        assert (tmp_path / f"beir_matrix_{dataset}.png").is_file()
        assert (tmp_path / f"beir_matrix_pareto_{dataset}.png").is_file()
        assert (tmp_path / f"beir_matrix_cost_pareto_{dataset}.png").is_file()
