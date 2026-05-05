from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _slugify(value: str) -> str:
    return "_".join(part for part in value.replace("-", " ").split()).lower()


def _load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Results CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Results CSV is empty: {path}")
    return df


def _plot_dataset(
    df: pd.DataFrame,
    *,
    output_path: Path,
    metric: str,
    title: str,
    dataset: str,
) -> None:
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    subset = df[df["dataset"] == dataset].copy()
    if subset.empty:
        raise ValueError(f"No rows found for dataset: {dataset}")
    subset = subset.sort_values("round")
    ax.plot(
        subset["round"],
        subset[metric],
        marker="o",
        linewidth=2,
        color="#4c78a8",
    )

    ax.set_title(title)
    ax.set_xlabel("Round")
    ax.set_ylabel(metric.replace("_", " ").upper())
    ax.grid(alpha=0.2)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        response = input(f"{output_path} exists. Overwrite? [y/N]: ").strip().lower()
        if response not in {"y", "yes"}:
            print("Canceled.")
            return
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot codegen replay results (MRR by round)."
    )
    parser.add_argument(
        "--input",
        default="results.csv",
        help="CSV produced by replay_codegen_rounds.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="assets",
        help="Directory to write plot image.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset to plot (e.g., msmarco).",
    )
    parser.add_argument(
        "--stat",
        choices=["mean", "median"],
        default="mean",
        help="Which summary statistic to plot.",
    )
    parser.add_argument(
        "--title",
        help="Override chart title.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    df = _load_results(input_path)

    dataset = args.dataset.strip()
    if not dataset:
        raise ValueError("Dataset is required.")

    metric_name = df["metric"].dropna().unique()
    if len(metric_name) != 1:
        raise ValueError(f"Expected a single metric column, found: {metric_name}")
    metric_key = metric_name[0].lower()
    column = f"{args.stat}_{metric_key}"
    if column not in df.columns:
        raise ValueError(f"Missing expected column: {column}")

    grouped = df.groupby(["dataset", "round"], as_index=False)[column].mean()
    title = args.title or f"Codegen replay {dataset} {column} by round"
    if args.title:
        filename = _slugify(args.title)
    else:
        filename = f"codegen_replay_{dataset}_{column}"
    output_path = Path(args.output_dir) / f"{filename}.png"

    _plot_dataset(
        grouped,
        output_path=output_path,
        metric=column,
        title=title,
        dataset=dataset,
    )
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
