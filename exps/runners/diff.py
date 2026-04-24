from __future__ import annotations

import numpy as np
import pandas as pd
from cheat_at_search.search import run_strategy
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.strategy_factory import create_strategy, load_strategy


class DiffParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_a_path: str
    strategy_b_path: str
    base_path: str | None = None
    dataset: Literal["esci", "msmarco", "wands"] = "wands"
    query: str | None = None
    k: int = 10
    num_queries: int | None = None
    seed: int = 42
    workers: int = 1
    sort: Literal["delta", "query"] = "delta"
    binary_relevance: str | None = None
    device: str | None = None
    no_cache: bool = False


class DiffResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_a_name: str
    strategy_b_name: str
    metric_name: str
    metric_a: pd.Series
    metric_b: pd.Series
    diff_table: pd.DataFrame
    query_results_a: pd.DataFrame | None = None
    query_results_b: pd.DataFrame | None = None
    query_metric_a: float | None = None
    query_metric_b: float | None = None


def _grade_column(judgments: pd.DataFrame) -> str | None:
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in judgments.columns:
            return col
    return None


def _query_text_map(judgments: pd.DataFrame) -> dict:
    if "query_id" in judgments.columns and "query" in judgments.columns:
        return (
            judgments[["query_id", "query"]]
            .drop_duplicates()
            .set_index("query_id")["query"]
            .to_dict()
        )
    return {}


def _metric_for_query(
    strategy, judgments: pd.DataFrame, query: str, metric_fn, *, cache: bool
) -> float | None:
    if "query" not in judgments.columns:
        return None
    subset = judgments[judgments["query"] == query]
    if subset.empty:
        return None
    graded = run_strategy(strategy, subset, cache=cache)
    series = metric_fn(graded)
    if series.empty:
        return None
    return float(series.iloc[0])


def _query_results(
    strategy,
    corpus: pd.DataFrame,
    judgments: pd.DataFrame,
    query: str,
    k: int,
) -> pd.DataFrame:
    top_k, scores = strategy.search(query, k=k)
    results = corpus.iloc[top_k].copy()
    results["score"] = scores
    grade_col = _grade_column(judgments)
    if grade_col and "query" in judgments.columns and "doc_id" in judgments.columns:
        match = judgments[judgments["query"] == query]
        grade_map = dict(zip(match["doc_id"], match[grade_col]))
        results["grade"] = results["doc_id"].map(grade_map)
    return results


def _diff_table(
    metric_name: str,
    metric_a: pd.Series,
    metric_b: pd.Series,
    judgments: pd.DataFrame,
    sort_by: str,
    name_a: str,
    name_b: str,
) -> pd.DataFrame:
    query_text = _query_text_map(judgments)
    metric_key = metric_name.lower()
    col_a = f"{metric_key}_{name_a}"
    col_b = f"{metric_key}_{name_b}"
    df_all = pd.DataFrame({col_a: metric_a, col_b: metric_b})
    df_all["diff"] = df_all[col_b] - df_all[col_a]
    df_all.index.name = "query_id"
    df_all["query"] = df_all.index.map(query_text.get)
    df_all = df_all.reset_index()
    df = df_all[df_all["diff"] != 0]

    if sort_by == "query":
        df = df.sort_values(by=["query", "diff"], ascending=[True, False])
    else:
        df = df.sort_values(by=["diff", "query"], ascending=[False, True])
    return df


def diff_benchmark(params: DiffParams) -> DiffResult:
    if params.seed is not None:
        np.random.seed(params.seed)

    strategy_a_config, params_a, requires_bm25_a = load_strategy(
        params.strategy_a_path, device=params.device, base_path=params.base_path
    )
    strategy_b_config, params_b, requires_bm25_b = load_strategy(
        params.strategy_b_path, device=params.device, base_path=params.base_path
    )
    requires_bm25 = requires_bm25_a or requires_bm25_b
    dataset = get_dataset(
        params.dataset, workers=params.workers, ensure_snowball=requires_bm25
    )
    corpus = dataset.corpus
    judgments = dataset.judgments
    metric_name, metric_fn = metric_for_dataset(params.dataset)
    strategy_a, _ = create_strategy(
        strategy_a_config,
        corpus=corpus,
        workers=params.workers,
        params=params_a,
        device=params.device,
        dataset=params.dataset,
    )
    strategy_b, _ = create_strategy(
        strategy_b_config,
        corpus=corpus,
        workers=params.workers,
        params=params_b,
        device=params.device,
        dataset=params.dataset,
    )

    query_results_a = None
    query_results_b = None
    query_metric_a = None
    query_metric_b = None
    if params.query:
        query_results_a = _query_results(
            strategy_a, corpus, judgments, params.query, params.k
        )
        query_results_b = _query_results(
            strategy_b, corpus, judgments, params.query, params.k
        )
        query_metric_a = _metric_for_query(
            strategy_a, judgments, params.query, metric_fn, cache=not params.no_cache
        )
        query_metric_b = _metric_for_query(
            strategy_b, judgments, params.query, metric_fn, cache=not params.no_cache
        )

    available_queries = judgments[["query", "query_id"]].drop_duplicates()
    if params.num_queries:
        available_queries = available_queries.sample(
            params.num_queries, random_state=params.seed
        )
    queries = available_queries["query"].tolist()
    graded_a = run_strategy(
        strategy_a,
        judgments,
        queries=queries,
        seed=params.seed,
        cache=not params.no_cache,
    )
    graded_b = run_strategy(
        strategy_b,
        judgments,
        queries=queries,
        seed=params.seed,
        cache=not params.no_cache,
    )
    metric_a = metric_fn(graded_a)
    metric_b = metric_fn(graded_b)
    diff_table = _diff_table(
        metric_name,
        metric_a,
        metric_b,
        judgments,
        params.sort,
        strategy_a_config.name,
        strategy_b_config.name,
    )

    return DiffResult(
        strategy_a_name=strategy_a_config.name,
        strategy_b_name=strategy_b_config.name,
        metric_name=metric_name,
        metric_a=metric_a,
        metric_b=metric_b,
        diff_table=diff_table,
        query_results_a=query_results_a,
        query_results_b=query_results_b,
        query_metric_a=query_metric_a,
        query_metric_b=query_metric_b,
    )
