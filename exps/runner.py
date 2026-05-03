import argparse
import csv
import json
import subprocess
from pathlib import Path

from exps.datasets import DATASET_NAMES
from exps.runners.run import RunParams, run_benchmark


def _report_metric(metric_name: str, metric_series, graded=None) -> None:
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


def _write_summary_csv(path: str, *, dataset: str, result) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        commit_sha = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit_sha = ""
    row = {
        "dataset": dataset,
        "commit_sha": commit_sha,
        "strategy_name": result.strategy_name,
        "strategy_params": json.dumps(result.strategy_params, sort_keys=True),
        "metric_name": result.metric_name,
        **result.summary,
    }

    file_exists = output_path.exists()
    fieldnames = None
    if file_exists:
        with output_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames

    if not fieldnames:
        fieldnames = list(row.keys())

    write_header = not file_exists or output_path.stat().st_size == 0
    with output_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


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
            binary_relevance=args.binary_relevance,
            device=args.device,
            no_cache=args.no_cache,
        )
        result = run_benchmark(params)
        if result.codegen_artifact_path:
            print(f"Codegen artifact: {result.codegen_artifact_path}")
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
        binary_relevance=args.binary_relevance,
        device=args.device,
        no_cache=args.no_cache,
    )
    result = run_benchmark(params)
    if result.codegen_artifact_path:
        print(f"Codegen artifact: {result.codegen_artifact_path}")
        print("")
    _report_metric(result.metric_name, result.metric_series, result.graded)
    if args.summary_csv:
        _write_summary_csv(args.summary_csv, dataset=args.dataset, result=result)


if __name__ == "__main__":
    main()
