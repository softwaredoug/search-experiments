import argparse
import csv
import json
import subprocess
import uuid
from pathlib import Path

import pandas as pd

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


def _current_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _append_csv_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = output_path.exists()
    fieldnames = None
    existing_rows = []
    if file_exists:
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            existing_rows = list(reader)

    if not fieldnames:
        fieldnames = list(rows[0].keys())
    else:
        new_fields = [
            key for row in rows for key in row if key not in fieldnames
        ]
        fieldnames.extend(dict.fromkeys(new_fields))
        if new_fields:
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)

    write_header = not file_exists or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _write_summary_csv(
    path: str,
    *,
    dataset: str,
    result,
    seed: int | None = None,
    run_id: str = "",
    commit_sha: str | None = None,
) -> None:
    row = {
        "dataset": dataset,
        "run_id": run_id,
        "commit_sha": commit_sha if commit_sha is not None else _current_commit_sha(),
        "seed": seed if seed is not None else "",
        "strategy_name": result.strategy_name,
        "strategy_params": json.dumps(result.strategy_params, sort_keys=True),
        "metric_name": result.metric_name,
        **result.summary,
    }
    _append_csv_rows(path, [row])


def _write_per_query_csv(
    path: str,
    *,
    dataset: str,
    seed: int,
    run_id: str,
    commit_sha: str,
    result,
) -> None:
    graded = result.graded
    if graded is None or graded.empty:
        raise ValueError("Per-query CSV output requires graded benchmark results.")
    query_keys = ["query_id", "query"]
    missing_query_keys = set(query_keys).difference(graded.columns)
    if missing_query_keys:
        raise ValueError(
            "Per-query results are missing query identifiers: "
            + ", ".join(sorted(missing_query_keys))
        )

    metric_columns = [
        column
        for column in (
            "ndcg_m",
            "ndcg_g",
            "max_ndcg_m",
            "mrr",
            "agreement",
            "pairwise_agreement",
        )
        if column in graded.columns
    ]
    resource_columns = sorted(
        column for column in graded.columns if column.startswith("resource.")
    )
    value_columns = [*metric_columns, *resource_columns]
    grouped = graded.groupby(query_keys, sort=False, dropna=False)
    for column in value_columns:
        if grouped[column].nunique(dropna=True).gt(1).any():
            raise ValueError(f"Per-query field {column!r} varies within a query.")
    per_query = grouped[value_columns].first().reset_index()

    metadata = {
        "run_id": run_id,
        "dataset": dataset,
        "commit_sha": commit_sha,
        "seed": seed,
        "strategy_name": result.strategy_name,
        "metric_name": result.metric_name or "",
    }
    rows = []
    for _, query_row in per_query.iterrows():
        row = {**metadata}
        row.update({key: query_row[key] for key in query_keys})
        for column in value_columns:
            value = query_row[column]
            row[column] = "" if pd.isna(value) else value
        rows.append(row)
    _append_csv_rows(path, rows)


def _write_full_results_parquet(
    path: str,
    *,
    dataset: str,
    seed: int,
    run_id: str,
    commit_sha: str,
    result,
) -> None:
    graded = result.graded
    if graded is None or graded.empty:
        raise ValueError("Full-results Parquet output requires graded benchmark results.")
    required_columns = {"query_id", "query", "doc_id", "rank"}
    missing_columns = required_columns.difference(graded.columns)
    if missing_columns:
        raise ValueError(
            "Full results are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    output = graded.loc[graded["rank"] <= 10].copy()
    output["doc_id"] = output["doc_id"].astype("string")
    output["dataset"] = dataset
    output["run_id"] = run_id
    output["commit_sha"] = commit_sha
    output["seed"] = seed
    output["strategy_name"] = result.strategy_name
    output["strategy_params"] = json.dumps(result.strategy_params, sort_keys=True)
    output["metric_name"] = result.metric_name or ""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_path, index=False, compression="zstd")


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
    parser.add_argument(
        "--per-query-csv",
        help="Write one row per query with NDCG, relevance, and resource metrics (appends if exists).",
    )
    parser.add_argument(
        "--full-results-parquet",
        help="Write all ranked document results (up to rank 10 per query) to Parquet.",
    )
    args = parser.parse_args()

    if args.query:
        if args.per_query_csv:
            parser.error("--per-query-csv is only available for full benchmark runs.")
        if args.full_results_parquet:
            parser.error(
                "--full-results-parquet is only available for full benchmark runs."
            )
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
    )
    result = run_benchmark(params)
    if result.codegen_artifact_path:
        print(f"Codegen artifact: {result.codegen_artifact_path}")
        print("")
    elif result.artifact_path:
        print(f"Strategy artifact: {result.artifact_path}")
        print("")
    _report_metric(result.metric_name, result.metric_series, result.summary)
    run_id = uuid.uuid4().hex
    commit_sha = _current_commit_sha()
    if args.summary_csv:
        _write_summary_csv(
            args.summary_csv,
            dataset=args.dataset,
            result=result,
            seed=args.seed,
            run_id=run_id,
            commit_sha=commit_sha,
        )
    if args.per_query_csv:
        _write_per_query_csv(
            args.per_query_csv,
            dataset=args.dataset,
            seed=args.seed,
            run_id=run_id,
            commit_sha=commit_sha,
            result=result,
        )
    if args.full_results_parquet:
        _write_full_results_parquet(
            args.full_results_parquet,
            dataset=args.dataset,
            seed=args.seed,
            run_id=run_id,
            commit_sha=commit_sha,
            result=result,
        )
        print(f"Full results Parquet: {args.full_results_parquet}")


if __name__ == "__main__":
    main()
