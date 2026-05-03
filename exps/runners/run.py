from __future__ import annotations

import statistics
from datetime import datetime

import pandas as pd
from cheat_at_search.search import run_strategy
from pydantic import BaseModel, ConfigDict
from exps.datasets import DatasetName, get_dataset
from exps.metrics import metric_for_dataset
from exps.strategy_factory import create_strategy, load_strategy
from exps.strategies.agentic import AgenticSearchStrategy
from exps.trace_utils import build_agentic_trace_root


class RunParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_path: str
    base_path: str | None = None
    dataset: DatasetName = "wands"
    num_queries: int | None = None
    query: str | None = None
    k: int = 10
    seed: int = 42
    workers: int = 1
    binary_relevance: str | None = None
    device: str | None = None
    no_cache: bool = False


class RunResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_name: str
    strategy_params: dict
    metric_name: str | None = None
    metric_series: pd.Series | None = None
    summary: dict[str, float] | None = None
    graded: pd.DataFrame | None = None
    query_results: pd.DataFrame | None = None
    query_grade_col: str | None = None
    most_relevant_row: pd.Series | None = None
    most_relevant_grade_col: str | None = None
    relevant_examples: list[dict] | None = None
    codegen_artifact_path: str | None = None


def _display_title(row: pd.Series) -> str:
    title = row.get("title", "")
    if isinstance(title, str) and title.strip():
        return title
    if title:
        return str(title)
    description = row.get("description", "")
    return description if isinstance(description, str) else str(description)


def _display_description(row: pd.Series) -> str:
    description = row.get("description", "")
    if isinstance(description, str):
        return description
    return str(description)


def _grade_column(judgments: pd.DataFrame) -> str | None:
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in judgments.columns:
            return col
    return None


def _most_relevant_row(
    *, judgments: pd.DataFrame, corpus: pd.DataFrame, query: str
) -> tuple[pd.Series | None, str | None]:
    if "query" not in judgments.columns:
        return None, None
    grade_col = _grade_column(judgments)
    if grade_col is None:
        return None, None
    subset = judgments[judgments["query"] == query]
    if subset.empty:
        return None, grade_col
    scores = pd.to_numeric(subset[grade_col], errors="coerce")
    if scores.notna().any():
        top_idx = scores.idxmax()
        top_row = subset.loc[top_idx]
    else:
        top_row = subset.iloc[0]
    top_row = top_row.copy()
    doc_id = top_row.get("doc_id")
    display_title = ""
    if doc_id is not None and "doc_id" in corpus.columns:
        match = corpus[corpus["doc_id"] == doc_id]
        if not match.empty:
            display_title = _display_title(match.iloc[0])
    top_row["display_title"] = display_title
    return top_row, grade_col


def _relevant_examples(
    *, judgments: pd.DataFrame, corpus: pd.DataFrame, query: str
) -> tuple[list[dict], str | None]:
    if "query" not in judgments.columns or "doc_id" not in judgments.columns:
        return [], None
    grade_col = _grade_column(judgments)
    if grade_col is None:
        return [], None
    subset = judgments[judgments["query"] == query]
    if subset.empty:
        return [], grade_col
    scores = pd.to_numeric(subset[grade_col], errors="coerce")
    if scores.notna().any():
        subset = subset.assign(_grade=scores).sort_values("_grade", ascending=False)
    if "doc_id" not in corpus.columns:
        return [], grade_col
    examples = []
    for _, row in subset.head(3).iterrows():
        doc_id = row.get("doc_id")
        grade = row.get(grade_col)
        match = corpus[corpus["doc_id"] == doc_id]
        if match.empty:
            continue
        item = match.iloc[0]
        examples.append(
            {
                "doc_id": doc_id,
                "grade": grade,
                "title": _display_title(item),
                "description": _display_description(item),
            }
        )
    return examples, grade_col


def _query_results(
    *,
    strategy,
    corpus: pd.DataFrame,
    judgments: pd.DataFrame,
    query: str,
    k: int,
) -> tuple[pd.DataFrame, str | None]:
    top_k, scores = strategy.search(query, k=k)
    results = corpus.iloc[top_k].copy()
    results["score"] = scores
    grade_col = _grade_column(judgments)
    if grade_col and "query" in judgments.columns and "doc_id" in judgments.columns:
        match = judgments[judgments["query"] == query]
        grade_map = dict(zip(match["doc_id"], match[grade_col]))
        results["grade"] = results["doc_id"].map(grade_map)
    results["display_title"] = results.apply(_display_title, axis=1)
    return results, grade_col


def run_benchmark(params: RunParams) -> RunResult:
    strategy_config, strategy_params, requires_bm25 = load_strategy(
        params.strategy_path, device=params.device, base_path=params.base_path
    )
    trace_path = None
    if strategy_config.type == "agentic":
        run_started_at = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trace_path = build_agentic_trace_root(
            strategy_config.name,
            params.dataset,
            run_started_at=run_started_at,
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
        dataset=params.dataset,
        trace_path=trace_path,
        judgments=judgments,
        report_num_queries=params.num_queries,
        report_seed=params.seed,
    )
    codegen_artifact_path = getattr(strategy, "artifact_path", None)
    codegen_artifact_path = (
        str(codegen_artifact_path) if codegen_artifact_path is not None else None
    )
    if params.query:
        query_results, grade_col = _query_results(
            strategy=strategy,
            corpus=corpus,
            judgments=judgments,
            query=params.query,
            k=params.k,
        )
        most_relevant_row, most_relevant_grade_col = _most_relevant_row(
            judgments=judgments, corpus=corpus, query=params.query
        )
        relevant_examples, relevant_grade_col = _relevant_examples(
            judgments=judgments, corpus=corpus, query=params.query
        )
        return RunResult(
            strategy_name=strategy_config.name,
            strategy_params=dict(strategy_config.params),
            query_results=query_results,
            query_grade_col=grade_col,
            most_relevant_row=most_relevant_row,
            most_relevant_grade_col=relevant_grade_col or most_relevant_grade_col,
            relevant_examples=relevant_examples,
            codegen_artifact_path=codegen_artifact_path,
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
    graded_queries = graded[["query", "query_id"]].drop_duplicates() if not graded.empty else pd.DataFrame()
    if len(graded_queries) != num_queries:
        raise ValueError(
            "Runner expected graded results for "
            f"{num_queries} queries but found {len(graded_queries)}."
        )
    metric_name, metric_fn = metric_for_dataset(params.dataset)
    metric_series = metric_fn(graded)
    if len(metric_series) != num_queries:
        raise ValueError(
            f"Runner expected {num_queries} metric values but found {len(metric_series)}."
        )
    metric_key = metric_name.lower()
    tool_calls = [1] * num_queries
    if isinstance(strategy, AgenticSearchStrategy):
        tool_calls = list(strategy.num_tool_calls.values())
        if not tool_calls:
            tool_calls = [0]
    summary = {
        f"mean_{metric_key}": float(metric_series.mean()) if not metric_series.empty else 0.0,
        f"median_{metric_key}": float(metric_series.median()) if not metric_series.empty else 0.0,
        "tool_calls_mean": float(statistics.fmean(tool_calls)),
        "tool_calls_median": float(statistics.median(tool_calls)),
        "tool_calls_std": float(statistics.pstdev(tool_calls)),
    }
    return RunResult(
        strategy_name=strategy_config.name,
        strategy_params=dict(strategy_config.params),
        metric_name=metric_name,
        metric_series=metric_series,
        summary=summary,
        graded=graded,
        codegen_artifact_path=codegen_artifact_path,
    )
