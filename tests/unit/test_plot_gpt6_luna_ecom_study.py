import csv
import sys
from pathlib import Path

import pytest
import yaml

from scripts.plot_gpt6_luna_ecom_study import (
    MODEL_BY_STRATEGY,
    STRATEGIES,
    _read_mean_token_cost_usd,
    main,
    plot_cost_pareto,
)


CONFIG_DIR = Path("configs/agentic/gpt-6-luna-study")


def test_study_configs_cover_baseline_and_four_agentic_model_tool_pairs():
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
                    "metric_name": "NDCG",
                    "mean_resource.input_tokens": 2_000_000,
                    "mean_resource.cached_input_tokens": 500_000,
                    "mean_resource.output_tokens": 1_000_000,
                }
            )

    costs = _read_mean_token_cost_usd(summary_path)

    assert costs[("wands", "agentic_gpt6_luna_bm25")] == pytest.approx(0.655)
    assert costs[("wands", "agentic_gpt5_mini_bm25")] == pytest.approx(2.3875)


def test_cost_pareto_omits_judged_ndcg_ceiling_line(tmp_path, monkeypatch):
    import matplotlib.axes

    summary_path = tmp_path / "summary.csv"
    fieldnames = [
        "dataset",
        "strategy_name",
        "metric_name",
        "mean_ndcg",
        "mean_max_ndcg",
        "mean_resource.input_tokens",
        "mean_resource.output_tokens",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, strategy in enumerate(MODEL_BY_STRATEGY):
            writer.writerow(
                {
                    "dataset": "wands",
                    "strategy_name": strategy,
                    "metric_name": "NDCG",
                    "mean_ndcg": 0.4 + index / 10,
                    "mean_max_ndcg": 0.8,
                    "mean_resource.input_tokens": 1000,
                    "mean_resource.output_tokens": 200,
                }
            )

    horizontal_lines = []
    original_axhline = matplotlib.axes.Axes.axhline

    def record_axhline(axis, y=0, *args, **kwargs):
        horizontal_lines.append(y)
        return original_axhline(axis, y, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", record_axhline)
    plot_cost_pareto(summary_path, tmp_path / "cost_pareto.png", "wands")

    assert horizontal_lines == []


def test_main_writes_retrieval_and_cost_pareto_plots_for_both_datasets(
    tmp_path, monkeypatch
):
    summary_path = tmp_path / "summary.csv"
    fieldnames = [
        "dataset",
        "strategy_name",
        "metric_name",
        "mean_ndcg",
        "mean_max_ndcg",
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
                writer.writerow(
                    {
                        "dataset": dataset,
                        "strategy_name": strategy,
                        "metric_name": "NDCG",
                        "mean_ndcg": 0.3 + index / 20,
                        "mean_max_ndcg": 0.8,
                        "mean_resource.input_tokens": 1000 if is_agentic else "",
                        "mean_resource.cached_input_tokens": 100 if is_agentic else "",
                        "mean_resource.output_tokens": 200 if is_agentic else "",
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
