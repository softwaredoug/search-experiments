import argparse

import pandas as pd
from cheat_at_search.search import run_strategy

from prf.datasets import bm25_params_for_dataset, get_dataset
from prf.metrics import metric_for_dataset
from prf.strategy_config import load_strategy_config, resolve_strategy_class


def _report_metric(metric_name: str, metric_series: pd.Series) -> None:
    metric_key = metric_name.lower()
    if metric_series.empty:
        print(f"No {metric_name} results to report.")
        return

    print(f"Per-query {metric_name}:")
    print(metric_series.to_string())
    print("")
    print("Summary:")
    print(f"mean_{metric_key}={metric_series.mean():.4f}")
    print(f"median_{metric_key}={metric_series.median():.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PRF lexical strategies.")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Path to strategy YAML config.",
    )
    parser.add_argument(
        "--dataset",
        choices=["esci", "msmarco", "wands"],
        default="wands",
        help="Dataset to run against.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries to sample for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for query sampling.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for indexing/search.",
    )
    parser.add_argument(
        "--binary-relevance",
        help=(
            "Comma-separated fields to use binary relevance in PRF "
            "(title, description, category)."
        ),
    )
    args = parser.parse_args()

    strategy_config = load_strategy_config(args.strategy)
    strategy_cls = resolve_strategy_class(strategy_config.type)
    params = dict(strategy_config.params)
    requires_bm25 = True
    if strategy_config.type == "agentic":
        tool_names = params.get("search_tools")
        if tool_names is not None:
            requires_bm25 = "bm25" in tool_names
    dataset = get_dataset(
        args.dataset, workers=args.workers, ensure_snowball=requires_bm25
    )
    corpus = dataset.corpus
    judgments = dataset.judgments
    bm25_k1, bm25_b = bm25_params_for_dataset(args.dataset)
    if strategy_config.type == "bm25":
        if "bm25_k1" not in params and "k1" not in params:
            params["bm25_k1"] = bm25_k1
        if "bm25_b" not in params and "b" not in params:
            params["bm25_b"] = bm25_b
    strategy = strategy_cls(
        corpus,
        workers=args.workers,
        **params,
    )
    graded = run_strategy(
        strategy,
        judgments,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    metric_name, metric_fn = metric_for_dataset(args.dataset)
    metric_series = metric_fn(graded)
    _report_metric(metric_name, metric_series)


if __name__ == "__main__":
    main()
