import pandas as pd
from cheat_at_search.search import ndcgs

try:
    from cheat_at_search.search import mrrs as _mrrs
except ImportError:  # pragma: no cover - fallback when dependency lacks MRR
    _mrrs = None


def metric_for_dataset(dataset_name: str):
    if dataset_name in {"msmarco", "minimarco"}:
        return "MRR", mrrs
    return "NDCG", ndcgs


def mrrs(graded: pd.DataFrame, queries: list[str] | None = None) -> pd.Series:
    if _mrrs is not None:
        return _mrrs(graded, queries)

    if graded.empty:
        return pd.Series(dtype=float)

    query_col = _query_column(graded)
    grade_col = _grade_column(graded)
    if query_col is None or grade_col is None:
        return pd.Series(dtype=float)

    df = graded.copy()
    df["_rel"] = pd.to_numeric(df[grade_col], errors="coerce").fillna(0)

    if "rank" in df.columns:
        df["_rank"] = pd.to_numeric(df["rank"], errors="coerce").fillna(0)
    elif "score" in df.columns:
        df["_rank"] = df.groupby(query_col)["score"].rank(
            ascending=False, method="first"
        )
    else:
        df["_rank"] = df.groupby(query_col).cumcount() + 1

    relevant = df[df["_rel"] > 0]
    all_queries = _queries_series(df, query_col, queries)
    if relevant.empty:
        result = pd.Series(0.0, index=all_queries)
        result.index.name = query_col
        return result

    min_rank = relevant.groupby(query_col)["_rank"].min()
    result = 1.0 / min_rank
    result = result.reindex(all_queries, fill_value=0.0)
    result.index.name = query_col
    return result


def _query_column(df: pd.DataFrame) -> str | None:
    for col in ("query_id", "query"):
        if col in df.columns:
            return col
    return None


def _grade_column(df: pd.DataFrame) -> str | None:
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in df.columns:
            return col
    return None


def _queries_series(
    df: pd.DataFrame,
    query_col: str,
    queries: list[str] | None,
) -> pd.Series:
    if queries is None:
        return df[query_col].drop_duplicates()
    return pd.Series(queries, name=query_col).drop_duplicates()
