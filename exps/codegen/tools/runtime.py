from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from cheat_at_search.search import run_strategy

from exps.codegen.strategy import CodeGenSearchStrategy
from exps.codegen.utils import build_id_lookup, load_rerank_fn, resolve_grade_column, resolve_id_column


def make_eval_guardrail(
    *,
    corpus,
    judgments: pd.DataFrame,
    tool_fns: list[callable],
    rerank_name: str,
    seed: int,
    num_queries: int,
    queries: list[str] | None = None,
    workers: int = 1,
) -> callable:
    def eval_guardrail(code: str) -> pd.Series:
        strategy = CodeGenSearchStrategy(
            corpus,
            search_fn=tool_fns[0],
            tool_fns=tool_fns,
            code=code,
            rerank_name=rerank_name,
            workers=workers,
        )
        results = run_strategy(
            strategy,
            judgments,
            queries=queries,
            num_queries=None if queries else num_queries,
            seed=seed,
            cache=False,
        )
        ndcgs = results.groupby("query")["ndcg"].mean()
        return ndcgs

    return eval_guardrail


def make_training_eval_fn(
    *,
    corpus,
    judgments: pd.DataFrame,
    tool_fns: list[callable],
    rerank_name: str,
    seed: int,
    num_queries: int,
    queries: list[str] | None = None,
    workers: int = 1,
) -> callable:
    return make_eval_guardrail(
        corpus=corpus,
        judgments=judgments,
        tool_fns=tool_fns,
        rerank_name=rerank_name,
        seed=seed,
        num_queries=num_queries,
        queries=queries,
        workers=workers,
    )


def make_eval_tools(
    *,
    corpus,
    judgments: pd.DataFrame,
    tool_fns: list[callable],
    rerank_name: str,
    code_path,
    seed: int,
    num_queries: int,
    queries: list[str] | None = None,
    workers: int = 1,
):
    id_col = resolve_id_column(corpus)
    lookup = build_id_lookup(corpus, id_col)
    grade_col = resolve_grade_column(judgments)

    def run_evals() -> dict[str, Any]:
        """Evaluate the reranker on a sample of queries."""
        code = code_path.read_text(encoding="utf-8")
        strategy = CodeGenSearchStrategy(
            corpus,
            search_fn=tool_fns[0],
            tool_fns=tool_fns,
            code=code,
            rerank_name=rerank_name,
            workers=workers,
        )
        results = run_strategy(
            strategy,
            judgments,
            queries=queries,
            num_queries=None if queries else num_queries,
            seed=seed,
            cache=False,
        )
        ndcgs = results.groupby("query")["ndcg"].mean()
        query_ndcgs = {str(k): float(v) for k, v in ndcgs.to_dict().items()}
        return {
            "mean_ndcg": float(ndcgs.mean()) if not ndcgs.empty else 0.0,
            "query_ndcgs": query_ndcgs,
        }

    def run_reranker(query: str, label: bool = False):
        """Run the reranker on a query.

        Set label=True to include human labels if available.
        """
        code = code_path.read_text(encoding="utf-8")
        rerank_fn = load_rerank_fn(code, rerank_name)
        doc_ids = rerank_fn(*tool_fns, query=query)
        scores = np.arange(len(doc_ids), 0, -1)
        results = []
        label_map = None
        if (
            label
            and grade_col
            and "query" in judgments.columns
            and id_col in judgments.columns
        ):
            subset = judgments[judgments["query"] == query]
            if not subset.empty:
                label_map = dict(zip(subset[id_col], subset[grade_col]))
        for doc_id, score in zip(doc_ids, scores):
            index = lookup.get(str(doc_id))
            if index is None:
                continue
            row = corpus.iloc[index]
            entry = {
                "doc_id": str(row.get(id_col)),
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "score": int(score),
            }
            if label_map is not None:
                grade = label_map.get(row.get(id_col))
                entry["grade"] = int(grade) if grade is not None else None
            results.append(entry)
        return results

    return run_evals, run_reranker
