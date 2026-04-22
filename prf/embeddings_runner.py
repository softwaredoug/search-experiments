import argparse

from prf.datasets import get_dataset
from prf.embeddings import cache_paths_for_corpus, load_or_create_embeddings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or load cached embeddings for a dataset."
    )
    parser.add_argument(
        "--dataset",
        choices=["esci", "msmarco", "wands"],
        default="wands",
        help="Dataset to build embeddings for.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes for indexing/search.",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, workers=args.workers, ensure_snowball=False)
    corpus = dataset.corpus
    embeddings = load_or_create_embeddings(corpus)
    emb_path, meta_path = cache_paths_for_corpus(corpus)

    print(f"Embeddings ready: {embeddings.shape[0]} docs, dim={embeddings.shape[1]}")
    print(f"Cache: {emb_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    main()
