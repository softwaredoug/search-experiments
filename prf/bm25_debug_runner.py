import argparse
import csv
import sys

import numpy as np

from cheat_at_search.tokenizers import snowball_tokenizer
from prf.datasets import get_dataset
from prf.strategies.bm25 import BM25Strategy
from searcharray.similarity import compute_idf
from prf.strategies.prf_rerank_terms import bm25_search_details


def _display_title(row) -> str:
    title = row.get("title", "")
    if isinstance(title, str) and title.strip():
        return title
    if title:
        return str(title)
    description = row.get("description", "")
    return description if isinstance(description, str) else str(description)


def _normalize_debug_terms(raw_terms: str, query_terms: list[str]) -> list[str]:
    debug_terms = []
    for raw_term in raw_terms.split(","):
        raw_term = raw_term.strip()
        if not raw_term:
            continue
        debug_tokens = snowball_tokenizer(raw_term)
        term = debug_tokens[0] if debug_tokens else raw_term
        if term in query_terms:
            debug_terms.append(term)
    return debug_terms


def _term_matches(index, fields: dict[str, float], term: str) -> np.ndarray:
    matches = np.zeros(len(index), dtype=bool)
    for field in fields:
        field_snowball = f"{field}_snowball"
        if field_snowball in index:
            term_match = index[field_snowball].array.score(term)
            matches |= term_match > 0
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BM25 and print debug columns.")
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
        "--debug-terms",
        help="Comma-separated query terms to show df columns.",
    )
    parser.add_argument(
        "--proportion",
        action="store_true",
        help="Include bm25/doc_weight proportions over top 10 results.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Output results as CSV to stdout.",
    )
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    corpus = dataset.corpus
    strategy = BM25Strategy(corpus)

    tokenized = snowball_tokenizer(args.query)
    fields = {
        "title": strategy.title_boost,
        "description": strategy.description_boost,
    }
    bm25_scores, doc_weight, num_matches, term_dfs = bm25_search_details(
        strategy.index,
        fields,
        tokenized,
    )

    debug_terms = []
    if args.debug_terms:
        debug_terms = _normalize_debug_terms(args.debug_terms, tokenized)

    top_k = np.argsort(-bm25_scores)[: args.k]
    results = corpus.iloc[top_k].copy()
    results["bm25_score"] = bm25_scores[top_k]
    results["doc_weight"] = doc_weight[top_k]
    results["num_matches"] = num_matches[top_k]
    if args.proportion:
        top_10 = np.argsort(-bm25_scores)[:10]
        bm25_sum = bm25_scores[top_10].sum()
        doc_weight_sum = doc_weight[top_10].sum()
        results["bm25_prop"] = results["bm25_score"] / bm25_sum if bm25_sum else 0.0
        results["doc_weight_prop"] = (
            results["doc_weight"] / doc_weight_sum if doc_weight_sum else 0.0
        )
    for term in debug_terms:
        df = term_dfs.get(term, 0)
        idf = compute_idf(len(corpus), df)
        term_match_mask = _term_matches(strategy.index, fields, term)
        results[f"{term}_df"] = np.where(term_match_mask[top_k], df, 0)
        results[f"{term}_idf"] = np.where(term_match_mask[top_k], idf, 0)

    columns = ["bm25_score", "doc_weight", "num_matches"]
    if args.proportion:
        columns.extend(["bm25_prop", "doc_weight_prop"])
    for term in debug_terms:
        columns.append(f"{term}_df")
        columns.append(f"{term}_idf")
    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(["doc_id", *columns, "title"])
        for _, row in results.iterrows():
            doc_id = row.get("doc_id", "")
            title = _display_title(row)
            values = [
                f"{row['bm25_score']:.4f}",
                f"{row['doc_weight']:.4f}",
                f"{row['num_matches']:.0f}",
            ]
            if args.proportion:
                values.append(f"{row['bm25_prop']:.6f}")
                values.append(f"{row['doc_weight_prop']:.6f}")
            for term in debug_terms:
                values.append(str(int(row[f"{term}_df"])))
                values.append(f"{row[f'{term}_idf']:.6f}")
            writer.writerow([doc_id, *values, title])
        total_bm25 = results["bm25_score"].sum()
        total_doc_weight = results["doc_weight"].sum()
        totals = [f"{total_bm25:.4f}", f"{total_doc_weight:.4f}", ""]
        if args.proportion:
            totals.extend(["", ""])
        for _ in debug_terms:
            totals.extend(["", ""])
        writer.writerow(["TOTAL", *totals, ""])
        return

    print(f"Query: {args.query}")
    header = "doc_id\t" + "\t".join(columns) + "\ttitle"
    print(header)

    for _, row in results.iterrows():
        doc_id = row.get("doc_id", "")
        title = _display_title(row)
        values = [
            f"{row['bm25_score']:.4f}",
            f"{row['doc_weight']:.4f}",
            f"{row['num_matches']:.0f}",
        ]
        if args.proportion:
            values.append(f"{row['bm25_prop']:.6f}")
            values.append(f"{row['doc_weight_prop']:.6f}")
        for term in debug_terms:
            values.append(str(int(row[f"{term}_df"])))
            values.append(f"{row[f'{term}_idf']:.6f}")
        values_str = "\t".join(values)
        print(f"{doc_id}\t{values_str}\t{title}")

    total_bm25 = results["bm25_score"].sum()
    total_doc_weight = results["doc_weight"].sum()
    print("")
    print(f"Sum bm25_score={total_bm25:.4f}\tdoc_weight={total_doc_weight:.4f}")


if __name__ == "__main__":
    main()
