import csv
import sys

from scripts.plot_beir_matrix import DATASET_STRATEGIES, DATASETS, main


def test_matrix_includes_baseline_gpt6_luna_and_sid_only():
    labels = {
        label
        for strategies in DATASET_STRATEGIES.values()
        for _, label in strategies
    }

    assert "BM25" in labels
    assert "GPT-6 Luna · BM25" in labels
    assert "GPT-6 Luna · BM25 + E5" in labels
    assert not any("GPT-5" in label for label in labels)
    assert {"SID-2 · BM25 · low", "SID-2 · BM25 · medium", "SID-2 · BM25 · high"}.issubset(
        labels
    )
    assert {
        "SID-2 · BM25 + E5 · low",
        "SID-2 · BM25 + E5 · medium",
        "SID-2 · BM25 + E5 · high",
    }.issubset(labels)


def test_matrix_uses_scifact_specific_configs_only_for_scifact():
    scifact_names = {name for name, _ in DATASET_STRATEGIES["scifact"]}
    trec_covid_names = {name for name, _ in DATASET_STRATEGIES["trec_covid"]}

    assert "agentic_scifact_gpt6_luna_bm25" in scifact_names
    assert "agentic_scifact_sid_bm25_e5_medium" in scifact_names
    assert "agentic_scifact_gpt5_mini_bm25" not in scifact_names
    assert "agentic_bm25_ecommerce_gpt6_luna" not in scifact_names
    assert "agentic_bm25_e5_ecommerce_gpt6_luna" not in scifact_names
    assert "agentic_bm25_ecommerce_gpt6_luna" in trec_covid_names
    assert "agentic_bm25_e5_ecommerce_gpt6_luna" in trec_covid_names
    assert "agentic_trec_covid_gpt6_luna_bm25" in trec_covid_names
    assert "agentic_trec_covid_sid_bm25_e5_medium" in trec_covid_names
    assert "agentic_scifact_gpt6_luna_bm25" not in trec_covid_names

    trec_covid_labels = {label for _, label in DATASET_STRATEGIES["trec_covid"]}
    assert "GPT-6 Luna · generic · BM25" in trec_covid_labels
    assert "GPT-6 Luna · TREC-COVID tuned · BM25" in trec_covid_labels


def test_main_writes_quality_and_pareto_plots(tmp_path, monkeypatch):
    summary_path = tmp_path / "summary.csv"
    fieldnames = [
        "dataset",
        "strategy_name",
        "metric_name",
        "mean_ndcg_m",
        "mean_max_ndcg_m",
        "mean_resource.input_tokens",
        "mean_resource.cached_input_tokens",
        "mean_resource.output_tokens",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in DATASETS:
            for index, (strategy, _) in enumerate(DATASET_STRATEGIES[dataset]):
                writer.writerow(
                    {
                        "dataset": dataset,
                        "strategy_name": strategy,
                        "metric_name": "NDCG_M",
                        "mean_ndcg_m": 0.2 + index / 20,
                        "mean_max_ndcg_m": 0.8 if dataset == "scifact" else 0.9,
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
