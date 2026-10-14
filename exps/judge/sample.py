from __future__ import annotations

import pandas as pd

from exps.judge.engines.common import grade_value


def sample_judgments(
    corpus: pd.DataFrame,
    judgments: pd.DataFrame,
    *,
    grade_column: str = "grade",
    sample_size: int = 10,
    seed: int = 42,
) -> dict[str, list[tuple[int, float]]]:
    """Sample judged corpus rows per query, stratified by relevance grade."""
    doc_indices = {
        doc_id: index
        for index, doc_id in enumerate(corpus["doc_id"].tolist())
        if not pd.isna(doc_id)
    }
    available = judgments.dropna(
        subset=["query", "doc_id", grade_column]
    ).reset_index(drop=True)
    samples = {str(query): [] for query in available["query"].drop_duplicates()}
    available = available.assign(
        corpus_idx=available["doc_id"].map(doc_indices)
    ).dropna(subset=["corpus_idx"])
    if available.empty:
        return samples

    available = available.assign(
        _grade=available[grade_column].map(grade_value)
    )
    num_unique_grades = min(available["_grade"].nunique(), 10)
    per_grade_sample = max(round(sample_size / num_unique_grades), 1)
    by_query_grade = available.groupby(
        ["query", "_grade"], sort=False
    ).apply(
        lambda group: group.sample(
            n=min(len(group), per_grade_sample), random_state=seed
        ),
        include_groups=False,
    ).droplevel(-1).reset_index()
    # Randomize document order too, so pairwise training sees both side orientations.
    by_query_grade = by_query_grade.sample(frac=1, random_state=seed)
    result = by_query_grade.groupby("query", sort=False).apply(
        lambda group: [
            (int(index), float(grade))
            for index, grade in group[["corpus_idx", "_grade"]].itertuples(
                index=False, name=None
            )
        ],
        include_groups=False,
    )
    samples.update(
        {
            str(query): rows for query, rows in result.to_dict().items()
        }
    )
    return samples
