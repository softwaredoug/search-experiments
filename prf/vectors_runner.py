import argparse
from cheat_at_search.tokenizers import snowball_tokenizer

from prf.datasets import bm25_params_for_dataset, get_dataset
from prf.strategies.prf_rerank import PRFRerankStrategy


def _display_title(row) -> str:
    title = row.get("title", "")
    if isinstance(title, str) and title.strip():
        return title
    if title:
        return str(title)
    description = row.get("description", "")
    return description if isinstance(description, str) else str(description)


def _format_vector(term_scores: dict[str, float]) -> str:
    if not term_scores:
        return ""
    sorted_terms = sorted(term_scores.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{term}={score:.4f}" for term, score in sorted_terms)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a single query and print RM3 vectors."
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
        "--dataset",
        choices=["esci", "msmarco", "wands"],
        default="wands",
        help="Dataset to run against.",
    )
    parser.add_argument(
        "--fields",
        default="title,description",
        help="Comma-separated RM3 fields (title, description, category).",
    )
    parser.add_argument(
        "--binary-relevance",
        help=(
            "Comma-separated fields to use binary relevance in PRF "
            "(title, description, category)."
        ),
    )
    parser.add_argument(
        "--debug-terms",
        help="Comma-separated terms to trace RM3 scoring details.",
    )
    args = parser.parse_args()

    rm3_fields = [field.strip() for field in args.fields.split(",") if field.strip()]
    debug_terms = []
    if args.debug_terms:
        for raw_term in args.debug_terms.split(","):
            raw_term = raw_term.strip()
            if not raw_term:
                continue
            debug_tokens = snowball_tokenizer(raw_term)
            debug_terms.append(debug_tokens[0] if debug_tokens else raw_term)
    debug_terms_set = set(debug_terms)

    dataset = get_dataset(args.dataset)
    corpus = dataset.corpus
    bm25_k1, bm25_b = bm25_params_for_dataset(args.dataset)
    strategy = PRFRerankStrategy(
        corpus,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        rm3_fields=rm3_fields,
        binary_relevance_fields=args.binary_relevance,
    )

    if debug_terms:
        doc_vectors, top_k, scores, debug_info = strategy.vectors(
            args.query, k=args.k, debug_terms=debug_terms_set
        )
    else:
        doc_vectors, top_k, scores = strategy.vectors(args.query, k=args.k)
    results = corpus.iloc[top_k].copy()
    results["score"] = scores

    print(f"Query: {args.query}")
    for row_index, row in results.iterrows():
        doc_id = row.get("doc_id", "")
        title = _display_title(row)
        score = row.get("score", 0)
        vector = _format_vector(doc_vectors.get(row_index, {}))
        print(f"{doc_id}\t{score:.4f}\t{title}")
        if vector:
            print(f"RM3: {vector}")

    if debug_terms:
        print("")
        print(f"Debug terms: {', '.join(debug_terms)}")
        for field in rm3_fields:
            field_info = debug_info.get(field, {})
            print(f"Field: {field}")
            if not field_info:
                print("No debug info available for these terms.")
                continue
            for term in debug_terms:
                term_info = field_info.get(term)
                if not term_info:
                    continue
                print(f"{term} term_importance: {term_info['term_importance']:.6f}")
                print(f"{term} PWC: {term_info['pwc']:.6f}")

            for label, vector_key in (
                ("RM3 raw", "rm3_raw"),
                ("After term importance", "rm3_after_importance"),
                ("After doc weights", "rm3_after_doc_weights"),
            ):
                print("------------------------------------")
                print(label)
                header = "doc_id\t" + "\t".join(debug_terms) + "\ttitle"
                print(header)
                for doc_index in top_k:
                    row = corpus.iloc[doc_index]
                    doc_id = row.get("doc_id", "")
                    title = _display_title(row)
                    scores = []
                    for term in debug_terms:
                        term_info = field_info.get(term)
                        if not term_info:
                            scores.append("n/a")
                            continue
                        vector = term_info[vector_key]
                        scores.append(f"{vector[doc_index]:.6f}")
                    scores_str = "\t".join(scores)
                    print(f"{doc_id}\t{scores_str}\t{title}")


if __name__ == "__main__":
    main()
