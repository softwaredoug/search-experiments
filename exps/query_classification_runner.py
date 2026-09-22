import argparse
import csv
from pathlib import Path

from exps.datasets import DATASET_NAMES
from exps.runners.query_classification import (
    QueryClassificationParams,
    evaluate_query_classification,
)
from exps.strategy_config import load_strategy_config


def _write_summary_csv(path: str, *, strategy: str, dataset: str, threshold: float, result) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluations = result.evaluations or {result.eval_as: result}
    strategy_name = load_strategy_config(strategy).name
    fieldnames = [
        "strategy",
        "dataset",
        "eval_as",
        "query_threshold",
        "queries",
        "queries_with_ground_truth",
        "mean_recall",
        "mean_jaccard",
        "coverage",
    ]
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for eval_as, evaluation in evaluations.items():
            per_query = evaluation.per_query
            writer.writerow(
                {
                    "strategy": strategy_name,
                    "dataset": dataset,
                    "eval_as": eval_as,
                    "query_threshold": threshold,
                    "queries": len(per_query),
                    "queries_with_ground_truth": int(
                        per_query["expected_categories"].map(bool).sum()
                    ),
                    "mean_recall": evaluation.mean_recall,
                    "mean_jaccard": evaluation.mean_jaccard,
                    "coverage": evaluation.coverage,
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate query-understanding enrichment against judgments."
    )
    parser.add_argument("--strategy", required=True, help="Path to strategy YAML config.")
    parser.add_argument(
        "--dataset", choices=DATASET_NAMES, default="wands", help="Dataset to evaluate."
    )
    parser.add_argument("--query", help="Evaluate one query instead of the full dataset.")
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of queries to evaluate; defaults to the full dataset.",
    )
    parser.add_argument(
        "--query-threshold",
        type=float,
        default=0.8,
        help="Minimum positive-label category proportion for ground truth.",
    )
    parser.add_argument(
        "--eval-as",
        default="direct",
        help=(
            "Evaluate directly or at taxonomy levels; comma-separate values, "
            "e.g. taxonomy[0],taxonomy[1],direct."
        ),
    )
    parser.add_argument(
        "--report",
        dest="report_path",
        help="Write a detailed enrichment evaluation report to this pickle file.",
    )
    parser.add_argument(
        "--summary-csv",
        help="Append aggregate evaluation statistics to this CSV file.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", help="Embedding device override.")
    parser.add_argument("--base-path", help="Base path for relative strategy config paths.")
    args = parser.parse_args()

    result = evaluate_query_classification(
        QueryClassificationParams(
            strategy_path=args.strategy,
            base_path=args.base_path,
            dataset=args.dataset,
            query=args.query,
            limit=args.limit,
            query_threshold=args.query_threshold,
            eval_as=args.eval_as,
            report_path=args.report_path,
            workers=args.workers,
            device=args.device,
        )
    )
    evaluations = result.evaluations or {result.eval_as: result}
    if args.summary_csv:
        _write_summary_csv(
            args.summary_csv,
            strategy=args.strategy,
            dataset=args.dataset,
            threshold=args.query_threshold,
            result=result,
        )
    for index, (eval_as, evaluation) in enumerate(evaluations.items()):
        if len(evaluations) > 1:
            if index:
                print()
            print(f"Eval as: {eval_as}")
        if args.query:
            row = evaluation.per_query.iloc[0]
            print(f"Query: {row['query']}")
            print(f"Expected categories: {row['expected_categories']}")
            print(f"Generated categories: {row['generated_categories']}")
            if row["expected_categories"]:
                print(f"Recall: {row['recall']:.4f}")
                print(f"Jaccard: {row['jaccard']:.4f}")
            else:
                print("Recall: unavailable (no ground truth categories)")
                print("Jaccard: unavailable (no ground truth categories)")
            continue

        print(f"Queries: {len(evaluation.per_query)}")
        print(
            "Queries with ground truth: "
            f"{evaluation.per_query['expected_categories'].map(bool).sum()}"
        )
        mean_recall = (
            f"{evaluation.mean_recall:.4f}"
            if evaluation.mean_recall is not None
            else "unavailable"
        )
        mean_jaccard = (
            f"{evaluation.mean_jaccard:.4f}"
            if evaluation.mean_jaccard is not None
            else "unavailable"
        )
        print(f"Mean recall: {mean_recall}")
        print(f"Mean Jaccard: {mean_jaccard}")
        print(f"Coverage: {evaluation.coverage:.4f}")


if __name__ == "__main__":
    main()
