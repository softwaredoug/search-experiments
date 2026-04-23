from __future__ import annotations

import pandas as pd
from cheat_at_search.search import run_strategy
from pydantic import BaseModel, ConfigDict
from typing_extensions import Literal

from exps.datasets import get_dataset
from exps.metrics import metric_for_dataset
from exps.strategy_factory import create_strategy, load_strategy


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


def run_benchmark(params: RunParams) -> RunResult:
    strategy_config, strategy_params, requires_bm25 = load_strategy(
        params.strategy_path, device=params.device
    )
    dataset = get_dataset(
        params.dataset, workers=params.workers, ensure_snowball=requires_bm25
    )
    corpus = dataset.corpus
    judgments = dataset.judgments
    strategy, _ = create_strategy(
        strategy_config,
        corpus=corpus,
        workers=params.workers,
        params=strategy_params,
        device=params.device,
    )
    available_queries = judgments[["query", "query_id"]].drop_duplicates()
    num_queries = params.num_queries or len(available_queries)
    graded = run_strategy(
        strategy,
        judgments,
        num_queries=num_queries,
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
