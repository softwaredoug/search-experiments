import csv
import sys
from pathlib import Path

import pytest
import yaml

from scripts.plot_gpt6_luna_ecom_study import (
    MODEL_BY_STRATEGY,
    STRATEGIES,
    _read_ndcg_runs,
    _read_mean_token_cost_usd,
    main,
    plot_cost_pareto,
    plot_retrieval,
)


CONFIG_DIR = Path("configs/agentic/gpt-6-luna-study")


def test_study_configs_cover_baseline_and_six_agentic_model_tool_pairs():
    config_paths = sorted(CONFIG_DIR.glob("*.yml"))
    configs = {
        yaml.safe_load(path.read_text(encoding="utf-8"))["strategy"]["name"]:
        yaml.safe_load(path.read_text(encoding="utf-8"))["strategy"]
        for path in config_paths
    }
    expected_names = {strategy for strategy, _ in STRATEGIES}

    assert set(configs) == expected_names
    assert configs["bm25"]["type"] == "bm25"
    assert set(MODEL_BY_STRATEGY) == expected_names - {"bm25"}
    for strategy, model in MODEL_BY_STRATEGY.items():
        params = configs[strategy]["params"]
        assert params["model"] == model
        assert {"bm25": None} in params["search_tools"]
        assert ({"e5_base_v2": None} in params["search_tools"]) == (
            "_e5" in strategy
        )

    tuned_prompt_strategies = {
        "agentic_gpt5_mini_bm25",
        "agentic_gpt5_mini_bm25_e5",
        "agentic_gpt6_luna_bm25",
        "agentic_gpt6_luna_bm25_e5",
    }
    tuned_prompts = {
        configs[name]["params"]["system_prompt"] for name in tuned_prompt_strategies
    }
    assert len(tuned_prompts) == 1

    shopper_prompt_strategies = {
        "agentic_gpt5_mini_bm25_e5_shopper_prompt",
        "agentic_gpt6_luna_bm25_e5_shopper_prompt",
    }
    shopper_prompts = {
        configs[name]["params"]["system_prompt"] for name in shopper_prompt_strategies
    }
    assert len(shopper_prompts) == 1
    assert "average shoppers ideal ranking" in shopper_prompts.pop()


def test_read_mean_token_cost_uses_gpt5_mini_and_luna_rates(tmp_path):
    summary_path = tmp_path / "summary.csv"
    fieldnames = [
        "dataset",
        "strategy_name",
        "metric_name",
        "mean_resource.input_tokens",
        "mean_resource.cached_input_tokens",
        "mean_resource.output_tokens",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for strategy in (
            "agentic_gpt5_mini_bm25",
            "agentic_gpt6_luna_bm25",
        ):
            writer.writerow(
                {
                    "dataset": "wands",
                    "strategy_name": strategy,
                    "metric_name": "NDCG_M",
                    "mean_resource.input_tokens": 2_000_000,
                    "mean_resource.cached_input_tokens": 500_000,
                    "mean_resource.output_tokens": 1_000_000,
                }
            )

    costs = _read_mean_token_cost_usd(summary_path)

    assert costs[("wands", "agentic_gpt6_luna_bm25")] == pytest.approx(0.655)
    assert costs[("wands", "agentic_gpt5_mini_bm25")] == pytest.approx(2.3875)


def test_read_ndcg_runs_preserves_each_seeded_run(tmp_path):
    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "strategy_name", "metric_name", "mean_ndcg_m"],
        )
        writer.writeheader()
        writer.writerows(
            {
                "dataset": "wands",
                "strategy_name": "agentic_gpt6_luna_bm25",
                "metric_name": "NDCG_M",
                "mean_ndcg_m": score,
            }
            for score in (0.5, 0.6, 0.7)
        )

    assert _read_ndcg_runs(summary_path)[("wands", "agentic_gpt6_luna_bm25")] == [
        0.5,
        0.6,
        0.7,
    ]


def test_retrieval_plot_connects_variant_means_and_shows_each_run(tmp_path, monkeypatch):
    import matplotlib.axes

    summary_path = tmp_path / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "strategy_name", "metric_name", "mean_ndcg_m"],
        )
        writer.writeheader()
        for strategy, scores in (
            ("bm25", (0.5, 0.6, 0.7)),
            ("agentic_gpt5_mini_bm25", (0.2, 0.3, 0.4)),
        ):
            for score in scores:
                writer.writerow(
                    {
                        "dataset": "wands",
                        "strategy_name": strategy,
                        "metric_name": "NDCG_M",
                        "mean_ndcg_m": score,
                    }
                )

    mean_line_values = []
    run_mark_counts = []
    original_plot = matplotlib.axes.Axes.plot
    original_scatter = matplotlib.axes.Axes.scatter

    def record_plot(axis, *args, **kwargs):
        lines = original_plot(axis, *args, **kwargs)
        if kwargs.get("label") == "Mean across runs":
            mean_line_values.extend(lines[0].get_ydata())
        return lines

    def record_scatter(axis, *args, **kwargs):
        points = original_scatter(axis, *args, **kwargs)
        if len(points.get_offsets()):
            run_mark_counts.append(len(points.get_offsets()))
        return points

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", record_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", record_scatter)
    plot_retrieval(summary_path, tmp_path / "retrieval.png", "wands")

    assert mean_line_values == pytest.approx([0.3, 0.6])
    assert run_mark_counts == [3, 3]


def test_cost_pareto_omits_judged_ndcg_ceiling_line(tmp_path, monkeypatch):
    import matplotlib.axes

    summary_path = tmp_path / "summary.csv"
    fieldnames = [
        "dataset",
        "strategy_name",
        "metric_name",
        "mean_ndcg_m",
        "mean_max_ndcg_m",
        "mean_resource.input_tokens",
        "mean_resource.output_tokens",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, strategy in enumerate(MODEL_BY_STRATEGY):
            for run_index in range(3):
                writer.writerow(
                    {
                        "dataset": "wands",
                        "strategy_name": strategy,
                        "metric_name": "NDCG_M",
                        "mean_ndcg_m": 0.4 + index / 10 + run_index / 100,
                        "mean_max_ndcg_m": 0.8,
                        "mean_resource.input_tokens": 1000 + run_index * 100,
                        "mean_resource.output_tokens": 200 + run_index * 50,
                    }
                )

    horizontal_lines = []
    scatter_labels = []
    original_axhline = matplotlib.axes.Axes.axhline
    original_scatter = matplotlib.axes.Axes.scatter

    def record_axhline(axis, y=0, *args, **kwargs):
        horizontal_lines.append(y)
        return original_axhline(axis, y, *args, **kwargs)

    def record_scatter(axis, *args, **kwargs):
        points = original_scatter(axis, *args, **kwargs)
        if len(points.get_offsets()):
            scatter_labels.append(kwargs.get("label"))
        return points

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", record_axhline)
    monkeypatch.setattr(matplotlib.axes.Axes, "scatter", record_scatter)
    plot_cost_pareto(summary_path, tmp_path / "cost_pareto.png", "wands")

    assert horizontal_lines == []
    assert len(scatter_labels) == 24
    assert scatter_labels.count("Individual runs") == 1
    assert scatter_labels.count("Variant means") == 1


def test_main_writes_retrieval_and_cost_pareto_plots_for_both_datasets(
    tmp_path, monkeypatch
):
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
        for dataset in ("wands", "esci"):
            for index, (strategy, _) in enumerate(STRATEGIES):
                is_agentic = strategy in MODEL_BY_STRATEGY
                for run_index in range(3):
                    writer.writerow(
                        {
                            "dataset": dataset,
                            "strategy_name": strategy,
                            "metric_name": "NDCG_M",
                            "mean_ndcg_m": 0.3 + index / 20 + run_index / 100,
                            "mean_max_ndcg_m": 0.8,
                            "mean_resource.input_tokens": 1000 + run_index
                            if is_agentic
                            else "",
                            "mean_resource.cached_input_tokens": 100
                            if is_agentic
                            else "",
                            "mean_resource.output_tokens": 200 + run_index
                            if is_agentic
                            else "",
                        }
                    )

    output_prefix = tmp_path / "plots" / "gpt6_luna_ecom_study"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_gpt6_luna_ecom_study.py",
            "--input",
            str(summary_path),
            "--output-prefix",
            str(output_prefix),
        ],
    )

    main()

    for dataset in ("wands", "esci"):
        assert (tmp_path / "plots" / f"gpt6_luna_ecom_study_{dataset}.png").is_file()
        assert (
            tmp_path / "plots" / f"gpt6_luna_ecom_study_cost_pareto_{dataset}.png"
        ).is_file()
