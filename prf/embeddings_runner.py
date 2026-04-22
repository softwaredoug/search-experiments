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
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Number of documents per embedding chunk.",
    )
    parser.add_argument(
        "--device",
        help="Embedding device override (e.g., mps, cpu).",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset, workers=args.workers, ensure_snowball=False)
    corpus = dataset.corpus
    embeddings = load_or_create_embeddings(
        corpus,
        device=args.device,
        chunk_size=args.chunk_size,
        show_progress=True,
    )
    manifest_path, chunk_path = cache_paths_for_corpus(corpus)

    print(f"Embeddings ready: {embeddings.shape[0]} docs, dim={embeddings.shape[1]}")
    chunk_pattern = chunk_path.name.replace("chunk_0.npy", "chunk_*.npy")
    print(f"Manifest: {manifest_path}")
    print(f"Chunks: {chunk_path.parent}/{chunk_pattern}")


if __name__ == "__main__":
    main()
