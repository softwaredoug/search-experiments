import argparse
from datetime import datetime

from exps.datasets import DATASET_NAMES, get_dataset
from exps.strategy_factory import create_strategy, load_strategy
from exps.trace_utils import build_agentic_trace_root


def _display_title(row) -> str:
    title = row.get("title", "")
    if isinstance(title, str) and title.strip():
        return title
    if title:
        return str(title)
    description = row.get("description", "")
    return description if isinstance(description, str) else str(description)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single query for debugging.")
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
        "--query",
        required=True,
        help="Query string to run.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to return.",
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
    args = parser.parse_args()

    strategy_config, params, requires_bm25 = load_strategy(
        args.strategy, device=args.device
    )
    trace_path = None
    if strategy_config.type == "agentic":
        run_started_at = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trace_path = build_agentic_trace_root(
            strategy_config.name,
            args.dataset,
            run_started_at=run_started_at,
        )
    dataset = get_dataset(args.dataset, ensure_snowball=requires_bm25)
    corpus = dataset.corpus
    judgments = dataset.judgments
    strategy, _ = create_strategy(
        strategy_config,
        corpus=corpus,
        workers=1,
        params=params,
        device=args.device,
        dataset=args.dataset,
        trace_path=trace_path,
        judgments=judgments,
    )

    top_k, scores = strategy.search(args.query, k=args.k)
    results = corpus.iloc[top_k].copy()
    results["score"] = scores

    print(f"Query: {args.query}")
    for _, row in results.iterrows():
        doc_id = row.get("doc_id", "")
        title = _display_title(row)
        score = row.get("score", 0)
        print(f"{doc_id}\t{score:.4f}\t{title}")


if __name__ == "__main__":
    main()
