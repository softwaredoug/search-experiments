import argparse

from exps.datasets import DATASET_NAMES
from exps.runners.query_classification import (
    QueryClassificationParams,
    evaluate_query_classification,
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
        help="Evaluate directly or at a taxonomy level, e.g. taxonomy[0].",
    )
    parser.add_argument(
        "--report",
        dest="report_path",
        help="Write a detailed enrichment evaluation report to this pickle file.",
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
    if args.query:
        row = result.per_query.iloc[0]
        print(f"Query: {row['query']}")
        print(f"Expected categories: {row['expected_categories']}")
        print(f"Generated categories: {row['generated_categories']}")
        if row["expected_categories"]:
            print(f"Recall: {row['recall']:.4f}")
            print(f"Jaccard: {row['jaccard']:.4f}")
        else:
            print("Recall: unavailable (no ground truth categories)")
            print("Jaccard: unavailable (no ground truth categories)")
        return

    print(f"Queries: {len(result.per_query)}")
    print(
        "Queries with ground truth: "
        f"{result.per_query['expected_categories'].map(bool).sum()}"
    )
    mean_recall = f"{result.mean_recall:.4f}" if result.mean_recall is not None else "unavailable"
    mean_jaccard = (
        f"{result.mean_jaccard:.4f}" if result.mean_jaccard is not None else "unavailable"
    )
    print(f"Mean recall: {mean_recall}")
    print(f"Mean Jaccard: {mean_jaccard}")
    print(f"Coverage: {result.coverage:.4f}")


if __name__ == "__main__":
    main()
