import argparse

from exps.datasets import DATASET_NAMES
from exps.runners.train import TrainParams, train_strategy, _write_summary_csv


def _print_pairwise_cv_metrics(metadata: dict) -> None:
    cross_validation = metadata.get("cross_validation", {})
    feature_metrics = cross_validation.get("feature_metrics", {})
    if not feature_metrics:
        return

    def display(value):
        return "n/a" if value is None else f"{value:.3f}"

    print("Cross-validation pairwise metrics (accuracy / precision / coverage):")
    print(
        "  aggregate model: "
        f"{display(cross_validation.get('mean_accuracy'))} / "
        f"{display(cross_validation.get('mean_precision'))} / "
        f"{display(cross_validation.get('mean_prediction_coverage'))}"
    )
    for feature_name, metrics in feature_metrics.items():
        print(
            f"  {feature_name}: "
            f"{display(metrics.get('mean_accuracy'))} / "
            f"{display(metrics.get('mean_precision'))} / "
            f"{display(metrics.get('mean_prediction_coverage'))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train experiment strategies.")
    parser.add_argument(
        "--strategy",
        required=True,
        help="Path to strategy YAML config.",
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_NAMES,
        default="wands",
        help="Dataset to train against.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Number of queries to sample for training eval.",
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
        "--device",
        help="Embedding device override (e.g., mps, cpu).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        help="Override the strategy's outer training-round budget.",
    )
    parser.add_argument(
        "--summary-csv",
        help="Write summary stats to CSV (appends if exists).",
    )
    args = parser.parse_args()

    params = TrainParams(
        strategy_path=args.strategy,
        dataset=args.dataset,
        num_queries=args.num_queries,
        seed=args.seed,
        workers=args.workers,
        device=args.device,
        rounds=args.rounds,
    )
    result = train_strategy(params)
    print(f"Training artifact: {result.artifact_path}")
    _print_pairwise_cv_metrics(result.metadata)
    if args.summary_csv:
        _write_summary_csv(args.summary_csv, dataset=args.dataset, result=result)


if __name__ == "__main__":
    main()
