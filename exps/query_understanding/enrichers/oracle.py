from __future__ import annotations

import hashlib
import json

import pandas as pd


def _grade_column(judgments: pd.DataFrame) -> str | None:
    for column in ("grade", "relevance", "rel", "label", "score"):
        if column in judgments.columns:
            return column
    return None


class OracleEnricher:
    """Return categories attached to maximally graded documents for each query."""

    def __init__(self, *, corpus, judgments: pd.DataFrame, field: str):
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

        grades = pd.to_numeric(judgments[grade_column], errors="coerce")
        max_grade = grades.max()
        if pd.isna(max_grade):
            raise ValueError("Oracle enrichment requires at least one numeric grade.")

        category_by_doc = dict(zip(corpus["doc_id"], corpus[field]))
        relevant = judgments.loc[grades == max_grade, ["query", "doc_id"]]
        self._categories_by_query: dict[str, list[str]] = {}
        for query, rows in relevant.groupby("query", sort=False):
            categories = []
            for doc_id in rows["doc_id"]:
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
            "categories_by_query": self._categories_by_query,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()


def make_oracle_enricher(*, corpus, judgments: pd.DataFrame, field: str):
    return OracleEnricher(corpus=corpus, judgments=judgments, field=field)
