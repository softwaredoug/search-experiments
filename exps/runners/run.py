from __future__ import annotations

from datetime import datetime

import pandas as pd
from cheat_at_search.search import ndcgs, run_strategy
from pydantic import BaseModel, ConfigDict
from exps.datasets import DatasetName, get_dataset
from exps.metrics import metric_for_dataset
from exps.reporting import write_benchmark_reports
from exps.strategy_factory import create_strategy, load_strategy
from exps.trace_utils import build_agentic_trace_root


class RunParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    strategy_path: str
    base_path: str | None = None
    dataset: DatasetName = "wands"
    codegen_run_round: int | None = None
    num_queries: int | None = None
    query: str | None = None
    k: int = 10
    seed: int = 42
    workers: int = 1
    batch_size: int | None = None
    binary_relevance: str | None = None
    device: str | None = None
    no_cache: bool = False
    summary_csv: str | None = None


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
    artifact_path: str | None = None
    run_id: str | None = None
    reports_path: str | None = None
    # Retained as a compatibility alias for codegen run consumers.
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


def _resource_values_by_query(graded: pd.DataFrame) -> dict[str, pd.Series]:
    resource_columns = [
        column for column in graded.columns if column.startswith("resource.")
    ]
    if not resource_columns:
        return {}
    query_keys = ["query", "query_id"]
    if not set(query_keys).issubset(graded.columns):
        raise ValueError("Resource results must include query and query_id columns.")

    grouped = graded.groupby(query_keys, sort=False)
    resources = {}
    for column in resource_columns:
        if grouped[column].nunique(dropna=True).gt(1).any():
            raise ValueError(f"Resource {column!r} must have one value per query.")
        values = grouped[column].first()
        resources[column] = pd.to_numeric(values, errors="raise").dropna()
    return resources


def _resource_summary(resources: dict[str, pd.Series]) -> dict[str, float]:
    summary = {}
    for column, values in resources.items():
        if values.empty:
            continue
        summary[f"mean_{column}"] = float(values.mean())
        summary[f"median_{column}"] = float(values.median())
        summary[f"std_{column}"] = float(values.std(ddof=0))
    return summary


def _mean_max_ndcg_m(graded: pd.DataFrame) -> float:
    if graded.empty:
        return 0.0
    if "max_ndcg_m" not in graded.columns:
        raise ValueError(
            "Search results are missing max_ndcg_m; update cheat-at-search."
        )
    query_keys = [column for column in ("query", "query_id") if column in graded]
    if not query_keys:
        raise ValueError(
            "Search results must include query identifiers for max_ndcg_m."
        )
    grouped = graded.groupby(query_keys, sort=False)["max_ndcg_m"]
    if grouped.nunique(dropna=True).gt(1).any():
        raise ValueError("max_ndcg_m must have one value per query.")
    values = pd.to_numeric(grouped.first(), errors="raise").dropna()
    return float(values.mean()) if not values.empty else 0.0


def _query_results(
    *,
    strategy,
    corpus: pd.DataFrame,
    judgments: pd.DataFrame,
    query: str,
    k: int,
) -> tuple[pd.DataFrame, str | None]:
    top_k, scores, resources = strategy._unpack_search_result(
        strategy.search(query, k=k)
    )
    results = corpus.iloc[top_k].copy()
    results["score"] = scores
    for name, value in resources.items():
        results[name] = value
    grade_col = _grade_column(judgments)
    if grade_col and "query" in judgments.columns and "doc_id" in judgments.columns:
        match = judgments[judgments["query"] == query]
        grade_map = dict(zip(match["doc_id"], match[grade_col]))
        results["grade"] = results["doc_id"].map(grade_map)
    results["display_title"] = results.apply(_display_title, axis=1)
    return results, grade_col


def _mark_judged_results(
    graded: pd.DataFrame, judgments: pd.DataFrame
) -> pd.DataFrame:
    keys = ["query_id", "query", "doc_id"]
    judged_docs = pd.MultiIndex.from_frame(judgments[keys].drop_duplicates())
    result_docs = pd.MultiIndex.from_frame(graded[keys])
    output = graded.copy()
    output["judged"] = result_docs.isin(judged_docs)
    return output


def run_benchmark(params: RunParams) -> RunResult:
    strategy_config, strategy_params, requires_bm25 = load_strategy(
        params.strategy_path, device=params.device, base_path=params.base_path
    )
    trace_path = None
    if strategy_config.type in {"agentic", "scatter_gather_wands"}:
        run_started_at = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trace_path = build_agentic_trace_root(
            strategy_config.name,
            params.dataset,
            run_started_at=run_started_at,
        )
        print(f"Agentic trace root: {trace_path}")
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
        codegen_run_round=params.codegen_run_round,
        no_cache=params.no_cache,
    )
    artifact_path = getattr(strategy, "artifact_path", None)
    artifact_path = str(artifact_path) if artifact_path is not None else None
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
            artifact_path=artifact_path,
            codegen_artifact_path=(
                artifact_path if strategy_config.type == "codegen" else None
            ),
        )
    available_queries = judgments[["query", "query_id"]].drop_duplicates()
    num_queries = params.num_queries or len(available_queries)
    batch_size = params.batch_size if params.batch_size is not None else 100
    graded = run_strategy(
        strategy,
        judgments,
        num_queries=num_queries,
        seed=params.seed,
        cache=not params.no_cache,
        batch_size=batch_size,
        include_agreement=True,
        include_pairwise_agreement=True,
    )
    graded = _mark_judged_results(graded, judgments)
    if trace_path is not None:
        query_dirs = (
            [path for path in trace_path.iterdir() if path.is_dir()]
            if trace_path.exists()
            else []
        )
        if not query_dirs:
            print(
                "No agentic trace folders were created. If this was a cached run, "
                "re-run with --no-cache to force tool execution."
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
    ndcg_g_series = ndcgs(graded, ideal="global")
    ndcg_m_series = ndcgs(graded, ideal="max")
    resource_values = _resource_values_by_query(graded)
    summary = {
        "mean_ndcg_g": float(ndcg_g_series.mean()) if not ndcg_g_series.empty else 0.0,
        "median_ndcg_g": float(ndcg_g_series.median()) if not ndcg_g_series.empty else 0.0,
        "mean_ndcg_m": float(ndcg_m_series.mean()) if not ndcg_m_series.empty else 0.0,
        "median_ndcg_m": float(ndcg_m_series.median()) if not ndcg_m_series.empty else 0.0,
        "mean_max_ndcg_m": _mean_max_ndcg_m(graded),
    }
    if metric_name == "MRR":
        summary["mean_mrr"] = float(metric_series.mean()) if not metric_series.empty else 0.0
        summary["median_mrr"] = float(metric_series.median()) if not metric_series.empty else 0.0
    summary.update(_resource_summary(resource_values))
    if not graded.empty:
        query_agreement = graded.groupby(["query", "query_id"], sort=False)
        summary["mean_agreement"] = float(
            query_agreement["agreement"].first().mean()
        )
        summary["mean_pairwise_agreement"] = float(
            query_agreement["pairwise_agreement"].first().mean()
        )
    else:
        summary["mean_agreement"] = 0.0
        summary["mean_pairwise_agreement"] = 0.0
    result = RunResult(
        strategy_name=strategy_config.name,
        strategy_params=dict(strategy_config.params),
        metric_name=metric_name,
        metric_series=metric_series,
        summary=summary,
        graded=graded,
        artifact_path=artifact_path,
        codegen_artifact_path=(
            artifact_path if strategy_config.type == "codegen" else None
        ),
    )
    reports = write_benchmark_reports(
        result,
        dataset=params.dataset,
        seed=params.seed,
        artifact_path=artifact_path,
        summary_csv=params.summary_csv,
    )
    result.run_id = reports.run_id
    result.reports_path = str(reports.reports_path)
    return result
