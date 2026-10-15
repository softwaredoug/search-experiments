from scripts.plot_agentic_pareto_tool_calls import _load_tool_calls


def test_load_tool_calls_reads_only_resource_summary():
    rows = [
        {
            "dataset": "esci",
            "strategy_name": "agentic_bm25_ecommerce_gpt5_mini",
            "metric_name": "NDCG",
            "mean_resource.tool_calls": "4",
            "tool_calls_mean": "99",
        },
        {
            "dataset": "wands",
            "strategy_name": "agentic_e5_ecommerce_gpt5_mini",
            "metric_name": "NDCG",
            "mean_resource.tool_calls": "",
            "tool_calls_mean": "2",
        },
    ]

    assert _load_tool_calls(rows) == {
        ("esci", "agentic_bm25_ecommerce_gpt5_mini"): 4.0,
    }
