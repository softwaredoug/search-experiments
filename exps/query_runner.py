import argparse

from exps.datasets import bm25_params_for_dataset, get_dataset
from exps.strategy_config import load_strategy_config, resolve_strategy_class


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
            "Comma-separated fields to treat as binary relevance "
            "(title, description, category)."
        ),
    )
    parser.add_argument(
        "--device",
        help="Embedding device override (e.g., mps, cpu).",
    )
    args = parser.parse_args()

    strategy_config = load_strategy_config(args.strategy)
    strategy_cls = resolve_strategy_class(strategy_config.type)
    params = dict(strategy_config.params)
    requires_bm25 = True
    if strategy_config.type == "agentic":
        tool_names = params.get("search_tools")
        if tool_names is not None:
            requires_bm25 = "bm25" in tool_names
        if args.device and "embeddings_device" not in params:
            if tool_names is None or "embeddings" in tool_names:
                params["embeddings_device"] = args.device
    if strategy_config.type == "embedding":
        requires_bm25 = False
        if args.device and "device" not in params:
            params["device"] = args.device
    dataset = get_dataset(args.dataset, ensure_snowball=requires_bm25)
    corpus = dataset.corpus
    bm25_k1, bm25_b = bm25_params_for_dataset(args.dataset)
    if strategy_config.type == "bm25":
        if "bm25_k1" not in params and "k1" not in params:
            params["bm25_k1"] = bm25_k1
        if "bm25_b" not in params and "b" not in params:
            params["bm25_b"] = bm25_b
    strategy = strategy_cls(
        corpus,
        **params,
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
