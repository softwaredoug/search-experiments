import argparse

from prf.runners.run import RunParams, run_benchmark


def _report_metric(metric_name: str, metric_series) -> None:
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
    parser.add_argument(
        "--device",
        help="Embedding device override (e.g., mps, cpu).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass cached strategy results.",
    )
    args = parser.parse_args()

    params = RunParams(
        strategy_path=args.strategy,
        dataset=args.dataset,
        num_queries=args.num_queries,
        seed=args.seed,
        workers=args.workers,
        binary_relevance=args.binary_relevance,
        device=args.device,
        no_cache=args.no_cache,
    )
    result = run_benchmark(params)
    _report_metric(result.metric_name, result.metric_series)


if __name__ == "__main__":
    main()
