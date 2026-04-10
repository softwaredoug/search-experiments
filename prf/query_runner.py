import argparse

from prf.datasets import get_dataset
from prf.runner import STRATEGIES


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
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    corpus = dataset.corpus
    strategy_cls = STRATEGIES[args.strategy]
    strategy = strategy_cls(corpus)

    top_k, scores = strategy.search(args.query, k=args.k)
    results = corpus.iloc[top_k].copy()
    results["score"] = scores

    print(f"Query: {args.query}")
    for _, row in results.iterrows():
        doc_id = row.get("doc_id", "")
        title = row.get("title", "")
        score = row.get("score", 0)
        print(f"{doc_id}\t{score:.4f}\t{title}")


if __name__ == "__main__":
    main()
