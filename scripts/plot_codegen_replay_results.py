from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

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


def _load_round_labels(run_path: Path) -> dict[int, str]:
    rounds_path = run_path / "rounds.jsonl"
    if not rounds_path.exists():
        return {}
    labels: dict[int, str] = {}
    for line in rounds_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        round_num = record.get("round")
        short_name = record.get("short_name")
        if isinstance(round_num, int) and isinstance(short_name, str) and short_name.strip():
            labels[round_num] = short_name.strip()
    return labels


def _plot_dataset(
    df: pd.DataFrame,
    *,
    output_path: Path,
    metric: str,
    title: str,
    dataset: str,
    round_labels: dict[int, str] | None = None,
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

    y_values = subset[metric].dropna().astype(float).tolist()
    if y_values:
        y_min = min(y_values)
        y_max = max(y_values)
        ax.set_ylim(y_min - 0.1, y_max + 0.1)

    if round_labels:
        annotations = []
        for _, row in subset.iterrows():
            round_num = row.get("round")
            value = row.get(metric)
            if not isinstance(round_num, int) or not isinstance(value, (int, float)):
                continue
            label = round_labels.get(round_num)
            if not label:
                continue
            annotations.append(
                ax.annotate(
                    label,
                    (round_num, value),
                    textcoords="offset points",
                    xytext=(0, 0),
                    ha="right",
                    va="top",
                    rotation=-45,
                    rotation_mode="anchor",
                )
            )
        if annotations:
            fig = plt.gcf()
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for annotation in annotations:
                bbox = annotation.get_window_extent(renderer=renderer)
                width_points = bbox.width * 72.0 / fig.dpi
                height_points = bbox.height * 72.0 / fig.dpi
                annotation.set_position((width_points, -height_points))

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
    parser.add_argument(
        "--run-path",
        help="Optional codegen run path with rounds.jsonl for labels.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    df = _load_results(input_path)

    dataset = args.dataset.strip()
    if not dataset:
        raise ValueError("Dataset is required.")

    run_path_input: Optional[str] = args.run_path
    if run_path_input is None:
        run_path_input = input(
            "Optional run path for round labels (press Enter to skip): "
        ).strip()
    round_labels: dict[int, str] | None = None
    if run_path_input:
        round_labels = _load_round_labels(Path(run_path_input).expanduser())

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
        round_labels=round_labels,
    )
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
