#!/usr/bin/env python3
"""Export query-level ESCI differences between GPT-6 Luna and SID-2 high."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from cheat_at_search.search import run_strategy

from exps.datasets import get_dataset
from exps.strategy_factory import create_strategy, load_strategy
from exps.trace_utils import build_agentic_trace_root


ROOT = Path(__file__).resolve().parents[1]
LEFT_CONFIG = "configs/ecom_agentic/agentic_ecom_2tools_e5_gpt6_luna.yml"
RIGHT_CONFIG = "configs/ecom_sid/agentic_ecom_sid_bm25_e5_base_v2_high.yml"
CSV_FIELDS = [
    "QueryID",
    "Query",
    "NDCGLHS",
    "NDCGRHS",
    "NDCGDiff",
    "MaxNDCG",
    "RankLHS",
    "DocIDLHS",
    "TitleLHS",
    "GradeLHS",
    "RankRHS",
    "DocIDRHS",
    "TitleRHS",
    "GradeRHS",
]


def _strategy(config_path: str, *, dataset, device: str, workers: int):
    config, params, _ = load_strategy(str(ROOT / config_path), device=device)
    trace_path = build_agentic_trace_root(
        config.name,
        "esci",
        run_started_at=(
            f"diff_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        ),
    )
    strategy, _ = create_strategy(
        config,
        corpus=dataset.corpus,
        workers=workers,
        params=params,
        device=device,
        dataset="esci",
        trace_path=trace_path,
        judgments=dataset.judgments,
    )
    return config.name, strategy


def _search_with_query_logging(strategy, label: str) -> None:
    search = strategy.search

    def logged_search(query: str, k: int = 10):
        print(f"Searching [{label}]: {query}", flush=True)
        return search(query, k)

    strategy.search = logged_search


def _sample_queries(judgments: pd.DataFrame, *, num_queries: int, seed: int):
    queries = judgments[["query", "query_id"]].drop_duplicates()
    if num_queries:
        queries = queries.sample(frac=1, random_state=seed).head(num_queries)
    return queries.reset_index(drop=True)


def _query_results(graded: pd.DataFrame) -> dict[int, pd.DataFrame]:
    return {
        int(query_id): rows.sort_values("rank").head(10).copy()
        for query_id, rows in graded.groupby("query_id", sort=False)
    }


def _query_metrics(graded: pd.DataFrame, column: str) -> dict[int, float]:
    per_query = graded.drop_duplicates("query_id").set_index("query_id")
    return {
        int(query_id): float(row[column])
        for query_id, row in per_query.iterrows()
    }


def _csv_value(value):
    if pd.isna(value):
        return ""
    return value


def _judgment_grade(grade_lookup: dict[tuple[str, str], object], query_id: int, row: dict):
    doc_id = row.get("doc_id")
    if pd.isna(doc_id):
        return "NA"
    return grade_lookup.get((str(query_id), str(doc_id)), "NA")


def _rank_rows(
    query_id: int,
    query: str,
    left: pd.DataFrame,
    right: pd.DataFrame,
    ndcg_left: float,
    ndcg_right: float,
    max_ndcg: float,
    grade_lookup: dict[tuple[str, str], object],
) -> list[dict]:
    left_rows = left.to_dict("records")
    right_rows = right.to_dict("records")
    rows = []
    for index in range(max(len(left_rows), len(right_rows))):
        lhs = left_rows[index] if index < len(left_rows) else {}
        rhs = right_rows[index] if index < len(right_rows) else {}
        rows.append(
            {
                "QueryID": query_id,
                "Query": query,
                "NDCGLHS": ndcg_left,
                "NDCGRHS": ndcg_right,
                "NDCGDiff": ndcg_right - ndcg_left,
                "MaxNDCG": max_ndcg,
                "RankLHS": _csv_value(lhs.get("rank")),
                "DocIDLHS": _csv_value(lhs.get("doc_id")),
                "TitleLHS": _csv_value(lhs.get("title")),
                "GradeLHS": _judgment_grade(grade_lookup, query_id, lhs),
                "RankRHS": _csv_value(rhs.get("rank")),
                "DocIDRHS": _csv_value(rhs.get("doc_id")),
                "TitleRHS": _csv_value(rhs.get("title")),
                "GradeRHS": _judgment_grade(grade_lookup, query_id, rhs),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "research/results/esci_gpt6_luna_sid2_high_diff.csv",
    )
    parser.add_argument(
        "--ndcg-threshold",
        type=float,
        default=0.1,
        help="Include a query when absolute NDCG difference meets this value.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=1000,
        help="Query sample size; 0 uses the full dataset (default: 1000).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    if args.ndcg_threshold < 0:
        parser.error("--ndcg-threshold must be nonnegative")
    if args.num_queries < 0:
        parser.error("--num-queries must be nonnegative")

    dataset = get_dataset("esci", workers=args.workers, ensure_snowball=True)
    queries = _sample_queries(
        dataset.judgments,
        num_queries=args.num_queries,
        seed=args.seed,
    )
    left_name, left_strategy = _strategy(
        LEFT_CONFIG,
        dataset=dataset,
        device=args.device,
        workers=args.workers,
    )
    right_name, right_strategy = _strategy(
        RIGHT_CONFIG,
        dataset=dataset,
        device=args.device,
        workers=args.workers,
    )
    _search_with_query_logging(left_strategy, left_name)
    _search_with_query_logging(right_strategy, right_name)

    run_args = {
        "judgments": dataset.judgments,
        "num_queries": args.num_queries or None,
        "seed": args.seed,
        "batch_size": 100,
        "show_progress": False,
        "cache": not args.no_cache,
    }
    graded_left = run_strategy(left_strategy, **run_args)
    graded_right = run_strategy(right_strategy, **run_args)
    left_results = _query_results(graded_left)
    right_results = _query_results(graded_right)
    left_metrics = _query_metrics(graded_left, "ndcg")
    right_metrics = _query_metrics(graded_right, "ndcg")
    left_max_ndcg = _query_metrics(graded_left, "max_ndcg")
    right_max_ndcg = _query_metrics(graded_right, "max_ndcg")
    grade_lookup = {
        (str(row.query_id), str(row.doc_id)): row.grade
        for row in dataset.judgments[["query_id", "doc_id", "grade"]].itertuples(
            index=False
        )
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    query_rows = 0
    result_rows = 0
    with args.output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for index, query_row in enumerate(queries.itertuples(index=False), start=1):
            query_id = int(query_row.query_id)
            query = str(query_row.query)
            ndcg_left = left_metrics.get(query_id, 0.0)
            ndcg_right = right_metrics.get(query_id, 0.0)
            max_ndcg = left_max_ndcg.get(query_id, 0.0)
            if max_ndcg != right_max_ndcg.get(query_id, 0.0):
                raise ValueError(
                    f"Max NDCG mismatch between variants for query {query_id}."
                )
            difference = ndcg_right - ndcg_left
            included = abs(difference) >= args.ndcg_threshold
            status = "INCLUDED" if included else "skipped"
            print(
                f"[{index}/{len(queries)}] {status} "
                f"| ΔNDCG (SID − GPT-6)={difference:+.4f} "
                f"| query_id={query_id} | {query}",
                flush=True,
            )
            if not included:
                continue
            query_rows += 1
            rows = _rank_rows(
                query_id,
                query,
                left_results.get(query_id, pd.DataFrame()),
                right_results.get(query_id, pd.DataFrame()),
                ndcg_left,
                ndcg_right,
                max_ndcg,
                grade_lookup,
            )
            writer.writerows(rows)
            result_rows += len(rows)

    print(
        f"Wrote {result_rows} ranked-result rows for {query_rows} queries "
        f"(absolute NDCG difference >= {args.ndcg_threshold:.3f}) to {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
