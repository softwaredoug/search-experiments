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
    report_path: str | None = None
    workers: int = 1
    device: str | None = None


@dataclass
class QueryClassificationResult:
    per_query: pd.DataFrame
    mean_recall: float | None
    mean_jaccard: float | None
    coverage: float
    eval_as: str = "direct"
    evaluations: dict[str, "QueryClassificationResult"] | None = None


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


def _parse_eval_as(eval_as: str) -> list[str]:
    values = [value.strip() for value in eval_as.split(",")]
    if not values or any(not value for value in values):
        raise ValueError("eval_as must contain one or more comma-separated values.")
    if len(values) != len(set(values)):
        raise ValueError("eval_as values must be unique.")
    for value in values:
        _taxonomy_level(value)
    return values


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


def _project_category(category: object, taxonomy_level: int) -> str | None:
    parts = str(category).split("/")
    return parts[taxonomy_level].strip() if taxonomy_level < len(parts) else None


def _eval_as_suffix(eval_as: str) -> str:
    if eval_as == "direct":
        return "direct"
    level = eval_as.removeprefix("taxonomy[").removesuffix("]")
    return f"taxonomy_{level}"


def _evaluate_as(
    *,
    eval_as: str,
    queries: list[str],
    judgments: pd.DataFrame,
    corpus: pd.DataFrame,
    category_field: str,
    threshold: float,
    predictions: dict[str, list[str]],
) -> QueryClassificationResult:
    taxonomy_level = _taxonomy_level(eval_as)
    expected = _ground_truth(
        queries=queries,
        judgments=judgments,
        corpus=corpus,
        category_field=category_field,
        threshold=threshold,
        taxonomy_level=taxonomy_level,
    )

    rows = []
    for query in queries:
        generated_categories = pd.Series(predictions[query], dtype="object")
        generated_categories = _project_categories(generated_categories, taxonomy_level)
        generated_categories = sorted(set(generated_categories.tolist()))
        expected_categories = expected[query]
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
        float(nonempty_truth["jaccard"].mean())
        if not nonempty_truth.empty
        else None
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
        eval_as=eval_as,
    )


def _write_report(
    *,
    path: str,
    judgments: pd.DataFrame,
    corpus: pd.DataFrame,
    category_field: str,
    queries: list[str],
    evaluations: dict[str, QueryClassificationResult],
    eval_as_values: list[str],
    predictions: dict[str, list[str]],
) -> None:
    report = judgments[judgments["query"].isin(queries)].copy()
    report_category_column = "__report_category"
    report = report.merge(
        corpus[["doc_id", category_field]].rename(
            columns={category_field: report_category_column}
        ),
        on="doc_id",
        how="left",
    )
    report[category_field] = report.pop(report_category_column)
    report["predicted_categories"] = report["query"].map(predictions)

    for eval_as in eval_as_values:
        evaluation = evaluations[eval_as]
        suffix = "" if len(eval_as_values) == 1 else f"_{_eval_as_suffix(eval_as)}"
        query_results = evaluation.per_query.set_index("query")
        for column in ("expected_categories", "generated_categories", "recall", "jaccard"):
            report[f"{column}{suffix}"] = report["query"].map(query_results[column])

        taxonomy_level = _taxonomy_level(eval_as)
        if taxonomy_level is not None:
            report[f"{category_field}_level_{taxonomy_level}"] = report[
                category_field
            ].map(
                lambda category: (
                    _project_category(category, taxonomy_level)
                    if pd.notna(category)
                    else None
                )
            )
            report[f"ground_truth_{category_field}_level_{taxonomy_level}"] = report[
                f"expected_categories{suffix}"
            ].map(list)
            report[f"predicted_categories_level_{taxonomy_level}"] = report[
                "predicted_categories"
            ].map(
                lambda categories: sorted(
                    {
                        projected
                        for category in categories or []
                        if (projected := _project_category(category, taxonomy_level))
                        is not None
                    }
                )
            )

    report.to_pickle(path)


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
    eval_as_values = _parse_eval_as(params.eval_as)

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
    predictions: dict[str, list[str]] = {}
    for query in tqdm(queries, desc="Enriching queries", unit="query"):
        predictions[query] = sorted(set(strategy.enrich(query)))

    evaluations = {
        eval_as: _evaluate_as(
            eval_as=eval_as,
            queries=queries,
            judgments=judgments,
            corpus=corpus,
            category_field=category_field,
            threshold=params.query_threshold,
            predictions=predictions,
        )
        for eval_as in eval_as_values
    }
    if params.report_path is not None:
        _write_report(
            path=params.report_path,
            judgments=judgments,
            corpus=corpus,
            category_field=category_field,
            queries=queries,
            evaluations=evaluations,
            eval_as_values=eval_as_values,
            predictions=predictions,
        )
    if len(evaluations) == 1:
        return next(iter(evaluations.values()))
    first = next(iter(evaluations.values()))
    return QueryClassificationResult(
        per_query=first.per_query,
        mean_recall=first.mean_recall,
        mean_jaccard=first.mean_jaccard,
        coverage=first.coverage,
        eval_as=first.eval_as,
        evaluations=evaluations,
    )
