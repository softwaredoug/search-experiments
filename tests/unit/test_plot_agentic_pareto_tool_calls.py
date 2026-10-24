from scripts.plot_agentic_pareto_tool_calls import _load_ndcg, _load_tool_calls


def test_load_ndcg_reads_ndcg_m_from_summary_csv():
    rows = [
        {
            "dataset": "esci",
            "strategy_name": "agentic_bm25_ecommerce_gpt5_mini",
            "metric_name": "NDCG_M",
            "mean_ndcg_m": "0.42",
        }
    ]

    assert _load_ndcg(rows) == {
        ("esci", "agentic_bm25_ecommerce_gpt5_mini"): 0.42,
    }


def test_load_tool_calls_reads_only_resource_summary():
    rows = [
        {
            "dataset": "esci",
            "strategy_name": "agentic_bm25_ecommerce_gpt5_mini",
            "metric_name": "NDCG_M",
            "mean_resource.tool_calls": "4",
            "tool_calls_mean": "99",
        },
        {
            "dataset": "wands",
            "strategy_name": "agentic_e5_ecommerce_gpt5_mini",
            "metric_name": "NDCG_M",
            "mean_resource.tool_calls": "",
            "tool_calls_mean": "2",
        },
    ]

    assert _load_tool_calls(rows) == {
        ("esci", "agentic_bm25_ecommerce_gpt5_mini"): 4.0,
    }
