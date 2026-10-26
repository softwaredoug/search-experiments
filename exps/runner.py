import argparse

from exps.datasets import DATASET_NAMES
from exps.runners.run import RunParams, run_benchmark


def _report_metric(metric_name: str, metric_series, summary: dict[str, float]) -> None:
    if metric_series.empty:
        print(f"No {metric_name} results to report.")
        return

    print(f"Per-query {metric_name}:")
    print(metric_series.to_string())
    print("")
    print("# Summary")
    print("Retrieval Metrics:")
    for name, value in summary.items():
        if not name.startswith("mean_resource.") and not name.startswith(
            "median_resource."
        ) and not name.startswith("std_resource."):
            label = (
                "mean_max_ndcg_m (judged ceiling)"
                if name == "mean_max_ndcg_m"
                else name
            )
            print(f"{label}: {value:.4f}")

    resource_metrics = [
        (name.replace("_resource.", "_", 1), value)
        for name, value in summary.items()
        if name.startswith(("mean_resource.", "median_resource.", "std_resource."))
    ]
    if resource_metrics:
        print("")
        print("# Resources")
        for name, value in resource_metrics:
            print(f"{name}: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment strategies.")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Path to strategy YAML config.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="wands",
        help="Dataset to run against.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries to sample for evaluation.",
    )
    parser.add_argument(
        "--query",
        help="Optional query string to show ranked results.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to show for --query.",
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
        "--batch-size",
        type=int,
        help="Batch size for run_strategy query processing.",
    )
    parser.add_argument(
        "--binary-relevance",
        help=(
            "Comma-separated fields to treat as binary relevance "
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
        help=(
            "Bypass cached strategy results (run_strategy only; does not affect "
            "BM25 indices or embeddings)."
        ),
    )
    parser.add_argument(
        "--summary-csv",
        help="Write summary stats to CSV (appends if exists).",
    )
    args = parser.parse_args()

    if args.query:
        params = RunParams(
            strategy_path=args.strategy,
            dataset=args.dataset,
            query=args.query,
            k=args.k,
            seed=args.seed,
            workers=args.workers,
            batch_size=args.batch_size,
            binary_relevance=args.binary_relevance,
            device=args.device,
            no_cache=args.no_cache,
        )
        result = run_benchmark(params)
        if result.codegen_artifact_path:
            print(f"Codegen artifact: {result.codegen_artifact_path}")
            print("")
        elif result.artifact_path:
            print(f"Strategy artifact: {result.artifact_path}")
            print("")
        if result.relevant_examples:
            print("Relevant examples:")
            for example in result.relevant_examples:
                doc_id = example.get("doc_id", "")
                grade = example.get("grade", "")
                title = example.get("title", "")
                description = example.get("description", "")
                print(f"{doc_id}\t{grade}\t{title}\t{description}")
            print("")

        if result.query_results is None:
            return
        print("Query results:")
        header_cols = ["score"]
        if result.query_grade_col and "grade" in result.query_results.columns:
            header_cols.append("grade")
        header = "doc_id\t" + "\t".join(header_cols) + "\ttitle"
        print(header)
        for _, row in result.query_results.iterrows():
            doc_id = row.get("doc_id", "")
            values = [f"{row.get('score', 0):.4f}"]
            if result.query_grade_col and "grade" in result.query_results.columns:
                values.append(str(row.get("grade", "")))
            title = row.get("display_title", "")
            print(f"{doc_id}\t" + "\t".join(values) + f"\t{title}")
        return

    params = RunParams(
        strategy_path=args.strategy,
        dataset=args.dataset,
        num_queries=args.num_queries,
        seed=args.seed,
        workers=args.workers,
        batch_size=args.batch_size,
        binary_relevance=args.binary_relevance,
        device=args.device,
        no_cache=args.no_cache,
        summary_csv=args.summary_csv,
    )
    result = run_benchmark(params)
    if result.codegen_artifact_path:
        print(f"Codegen artifact: {result.codegen_artifact_path}")
        print("")
    elif result.artifact_path:
        print(f"Strategy artifact: {result.artifact_path}")
        print("")
    _report_metric(result.metric_name, result.metric_series, result.summary)
    if result.reports_path:
        print(f"Reports: {result.reports_path}")


if __name__ == "__main__":
    main()
