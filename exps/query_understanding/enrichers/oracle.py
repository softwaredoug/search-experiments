from __future__ import annotations

import hashlib
import json
import math

import pandas as pd


def _grade_column(judgments: pd.DataFrame) -> str | None:
    for column in ("grade", "relevance", "rel", "label", "score"):
        if column in judgments.columns:
            return column
    return None


class OracleEnricher:
    """Return top-grade categories, with a bounded lower-grade fallback."""

    def __init__(
        self,
        *,
        corpus,
        judgments: pd.DataFrame,
        field: str,
        max_grade_dist: float = 0,
    ):
        if "doc_id" not in corpus.columns:
            raise ValueError("Oracle enrichment requires a corpus doc_id column.")
        if field not in corpus.columns:
            raise ValueError(f"Oracle enrichment requires corpus field {field!r}.")
        missing = {"query", "doc_id"} - set(judgments.columns)
        if missing:
            raise ValueError(
                f"Oracle enrichment judgments missing required columns: {sorted(missing)}"
            )
        grade_column = _grade_column(judgments)
        if grade_column is None:
            raise ValueError("Oracle enrichment judgments require a grade column.")
        if isinstance(max_grade_dist, bool):
            raise ValueError("oracle.params.max_grade_dist must be non-negative.")
        try:
            max_grade_dist = float(max_grade_dist)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "oracle.params.max_grade_dist must be a non-negative number."
            ) from exc
        if not math.isfinite(max_grade_dist) or max_grade_dist < 0:
            raise ValueError(
                "oracle.params.max_grade_dist must be a non-negative number."
            )
        self.max_grade_dist = max_grade_dist

        grades = pd.to_numeric(judgments[grade_column], errors="coerce")
        max_grade = grades.max()
        if pd.isna(max_grade):
            raise ValueError("Oracle enrichment requires at least one numeric grade.")

        category_by_doc = dict(zip(corpus["doc_id"], corpus[field]))
        query_judgments = judgments[["query", "doc_id"]].copy()
        query_judgments["_grade"] = grades
        self._categories_by_query: dict[str, list[str]] = {}
        min_accepted_grade = max_grade - max_grade_dist
        for query, rows in query_judgments.groupby("query", sort=False):
            candidates = rows.loc[
                rows["_grade"].between(min_accepted_grade, max_grade), "_grade"
            ]
            if candidates.empty:
                continue
            selected_grade = candidates.max()
            relevant_doc_ids = rows.loc[
                rows["_grade"] == selected_grade, "doc_id"
            ]
            categories = []
            for doc_id in relevant_doc_ids:
                category = category_by_doc.get(doc_id)
                if pd.isna(category):
                    continue
                category = str(category)
                if not category.strip() or category in categories:
                    continue
                categories.append(category)
            self._categories_by_query[str(query)] = categories

    def enrich(self, query: str) -> list[str]:
        return list(self._categories_by_query.get(query, []))

    @property
    def cache_key(self) -> str:
        payload = {
            "type": "oracle",
            "max_grade_dist": self.max_grade_dist,
            "categories_by_query": self._categories_by_query,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()


def make_oracle_enricher(
    *,
    corpus,
    judgments: pd.DataFrame,
    field: str,
    max_grade_dist: float = 0,
):
    return OracleEnricher(
        corpus=corpus,
        judgments=judgments,
        field=field,
        max_grade_dist=max_grade_dist,
    )
