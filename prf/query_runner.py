import argparse

from prf.datasets import get_dataset
from prf.runner import STRATEGIES


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
        choices=sorted(STRATEGIES.keys()),
        help="Strategy to run.",
    )
    parser.add_argument(
        "--dataset",
        choices=["esci", "msmarco", "wands"],
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
            "Comma-separated fields to use binary relevance in PRF "
            "(title, description, category)."
        ),
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    corpus = dataset.corpus
    strategy_cls = STRATEGIES[args.strategy]
    if args.strategy == "prf_rerank":
        strategy = strategy_cls(
            corpus,
            binary_relevance_fields=args.binary_relevance,
        )
    else:
        strategy = strategy_cls(corpus)

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
