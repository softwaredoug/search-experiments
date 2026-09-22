from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd
from tqdm import tqdm

from exps.datasets import DatasetName, get_dataset
from exps.strategy_factory import create_strategy, load_strategy


@dataclass(frozen=True)
class QueryClassificationParams:
    strategy_path: str
    base_path: str | None = None
    dataset: DatasetName = "wands"
    query: str | None = None
    limit: int | None = None
    query_threshold: float = 0.8
    eval_as: str = "direct"
    workers: int = 1
    device: str | None = None


@dataclass
class QueryClassificationResult:
    per_query: pd.DataFrame
    mean_recall: float | None
    mean_jaccard: float | None
    coverage: float


def _grade_column(judgments: pd.DataFrame) -> str | None:
    for column in ("grade", "relevance", "rel", "label", "score"):
        if column in judgments.columns:
            return column
    return None


def _taxonomy_level(eval_as: str) -> int | None:
    if eval_as == "direct":
        return None
    match = re.fullmatch(r"taxonomy\[(\d+)\]", eval_as)
    if match is None:
        raise ValueError(
            "eval_as must be 'direct' or a taxonomy level such as 'taxonomy[0]'."
        )
    return int(match.group(1))


def _project_categories(categories: pd.Series, taxonomy_level: int | None) -> pd.Series:
    if taxonomy_level is None:
        return categories
    return categories.map(
        lambda category: (
            str(category).split("/")[taxonomy_level].strip()
            if taxonomy_level < len(str(category).split("/"))
            else None
        )
    ).dropna()


def _ground_truth(
    *,
    queries: list[str],
    judgments: pd.DataFrame,
    corpus: pd.DataFrame,
    category_field: str,
    threshold: float,
    taxonomy_level: int | None = None,
) -> dict[str, list[str]]:
    grade_column = _grade_column(judgments)
    required_judgment_columns = {"query", "doc_id"}
    if not required_judgment_columns.issubset(judgments.columns):
        missing = sorted(required_judgment_columns - set(judgments.columns))
        raise ValueError(f"Judgments missing required columns: {missing}")
    if grade_column is None:
        raise ValueError("Judgments require a relevance/grade column.")
    numeric_grades = pd.to_numeric(judgments[grade_column], errors="coerce")
    max_grade = numeric_grades.max()
    if pd.isna(max_grade):
        raise ValueError("Judgments require at least one numeric relevance grade.")
    if "doc_id" not in corpus.columns:
        raise ValueError("Corpus requires a doc_id column for classification evaluation.")
    if category_field not in corpus.columns:
        raise ValueError(f"Corpus missing category field: {category_field}")

    category_by_doc = corpus.set_index("doc_id")[category_field]
    truth: dict[str, list[str]] = {}
    for query in queries:
        query_rows = judgments[judgments["query"] == query]
        grades = pd.to_numeric(query_rows[grade_column], errors="coerce").fillna(0)
        positive_rows = query_rows.loc[grades == max_grade]
        if positive_rows.empty:
            truth[query] = []
            continue

        categories = positive_rows["doc_id"].map(category_by_doc).dropna().astype(str)
        categories = categories[categories != ""]
        categories = _project_categories(categories, taxonomy_level)
        categories = categories[categories != ""]
        if categories.empty:
            truth[query] = []
            continue
        proportions = categories.value_counts(normalize=True)
        truth[query] = sorted(
            category for category, proportion in proportions.items() if proportion >= threshold
        )
    return truth


def evaluate_query_classification(
    params: QueryClassificationParams,
) -> QueryClassificationResult:
    if not 0 <= params.query_threshold <= 1:
        raise ValueError("query_threshold must be between 0 and 1.")
    if params.limit is not None and params.limit <= 0:
        raise ValueError("limit must be greater than 0.")
    taxonomy_level = _taxonomy_level(params.eval_as)

    strategy_config, strategy_params, requires_bm25 = load_strategy(
        params.strategy_path,
        device=params.device,
        base_path=params.base_path,
    )
    if strategy_config.type != "query_understanding":
        raise ValueError(
            "query_classification requires a query_understanding strategy; "
            f"received {strategy_config.type!r}."
        )

    dataset = get_dataset(
        params.dataset,
        workers=params.workers,
        ensure_snowball=requires_bm25,
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
    )

    if not hasattr(strategy, "enrich"):
        raise ValueError("query_understanding strategy must expose enrich(query).")
    category_field = strategy.category_field
    queries = [params.query] if params.query is not None else list(
        judgments["query"].drop_duplicates()
    )
    if params.query is None and params.limit is not None:
        queries = queries[: params.limit]
    expected = _ground_truth(
        queries=queries,
        judgments=judgments,
        corpus=corpus,
        category_field=category_field,
        threshold=params.query_threshold,
        taxonomy_level=taxonomy_level,
    )

    enriched_queries = []
    for query in tqdm(queries, desc="Enriching queries", unit="query"):
        generated_categories = pd.Series(strategy.enrich(query), dtype="object")
        generated_categories = _project_categories(generated_categories, taxonomy_level)
        enriched_queries.append(
            (query, expected[query], sorted(set(generated_categories.tolist())))
        )

    rows = []
    for query, expected_categories, generated_categories in enriched_queries:
        expected_set = set(expected_categories)
        generated_set = set(generated_categories)
        intersection = expected_set & generated_set
        union = expected_set | generated_set
        recall = (
            len(intersection) / len(expected_set)
            if expected_set
            else float("nan")
        )
        jaccard = len(intersection) / len(union) if expected_set else float("nan")
        rows.append(
            {
                "query": query,
                "expected_categories": expected_categories,
                "generated_categories": generated_categories,
                "recall": recall,
                "jaccard": jaccard,
            }
        )

    per_query = pd.DataFrame(rows)
    nonempty_truth = per_query[per_query["expected_categories"].map(bool)]
    mean_recall = float(nonempty_truth["recall"].mean()) if not nonempty_truth.empty else None
    mean_jaccard = (
        float(nonempty_truth["jaccard"].mean()) if not nonempty_truth.empty else None
    )
    coverage = (
        float(per_query["generated_categories"].map(bool).mean())
        if not per_query.empty
        else 0.0
    )
    return QueryClassificationResult(
        per_query=per_query,
        mean_recall=mean_recall,
        mean_jaccard=mean_jaccard,
        coverage=coverage,
    )
