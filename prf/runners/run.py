from __future__ import annotations

import pandas as pd
from cheat_at_search.search import run_strategy
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from prf.datasets import bm25_params_for_dataset, get_dataset
from prf.metrics import metric_for_dataset
from prf.strategy_config import load_strategy_config, resolve_strategy_class


class RunParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_path: str
    dataset: Literal["esci", "msmarco", "wands"] = "wands"
    num_queries: int | None = None
    seed: int = 42
    workers: int = 1
    binary_relevance: str | None = None
    device: str | None = None
    no_cache: bool = False


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_name: str
    metric_name: str
    metric_series: pd.Series
    summary: dict[str, float]
    graded: pd.DataFrame | None = None


def _requires_bm25(strategy_type: str, params: dict) -> bool:
    if strategy_type == "bm25":
        return True
    if strategy_type == "embedding":
        return False
    if strategy_type == "agentic":
        tool_names = params.get("search_tools")
        if tool_names is None:
            return True
        return "bm25" in tool_names
    return True


def run_benchmark(params: RunParams) -> RunResult:
    strategy_config = load_strategy_config(params.strategy_path)
    strategy_cls = resolve_strategy_class(strategy_config.type)
    strategy_params = dict(strategy_config.params)
    requires_bm25 = _requires_bm25(strategy_config.type, strategy_params)

    if params.device:
        if strategy_config.type == "agentic" and "embeddings_device" not in strategy_params:
            tool_names = strategy_params.get("search_tools")
            if tool_names is None or "embeddings" in tool_names:
                strategy_params["embeddings_device"] = params.device
        if strategy_config.type == "embedding" and "device" not in strategy_params:
            strategy_params["device"] = params.device

    dataset = get_dataset(
        params.dataset, workers=params.workers, ensure_snowball=requires_bm25
    )
    corpus = dataset.corpus
    judgments = dataset.judgments
    bm25_k1, bm25_b = bm25_params_for_dataset(params.dataset)

    if strategy_config.type == "bm25":
        if "bm25_k1" not in strategy_params and "k1" not in strategy_params:
            strategy_params["bm25_k1"] = bm25_k1
        if "bm25_b" not in strategy_params and "b" not in strategy_params:
            strategy_params["bm25_b"] = bm25_b

    strategy = strategy_cls(
        corpus,
        workers=params.workers,
        **strategy_params,
    )
    graded = run_strategy(
        strategy,
        judgments,
        num_queries=params.num_queries,
        seed=params.seed,
        cache=not params.no_cache,
    )
    metric_name, metric_fn = metric_for_dataset(params.dataset)
    metric_series = metric_fn(graded)
    if metric_series.index.name != "query_id" and "query_id" in graded.columns:
        if "query" in graded.columns:
            query_map = (
                graded[["query", "query_id"]]
                .drop_duplicates()
                .set_index("query")["query_id"]
            )
            metric_series = metric_series.copy()
            metric_series.index = metric_series.index.map(query_map.get)
            metric_series.index.name = "query_id"
    metric_key = metric_name.lower()
    summary = {
        f"mean_{metric_key}": float(metric_series.mean()) if not metric_series.empty else 0.0,
        f"median_{metric_key}": float(metric_series.median()) if not metric_series.empty else 0.0,
    }
    return RunResult(
        strategy_name=strategy_config.name,
        metric_name=metric_name,
        metric_series=metric_series,
        summary=summary,
        graded=graded,
    )
