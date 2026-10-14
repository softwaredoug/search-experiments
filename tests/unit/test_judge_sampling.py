from __future__ import annotations

import pandas as pd

from exps.judge.sample import sample_judgments


def test_sample_judgments_supports_custom_grade_column_and_unmatched_docs():
    grades = [0.0, 0.0, 0.1, 0.1, 0.5, 0.5, 1.0, 1.0]
    corpus = pd.DataFrame(
        {
            "doc_id": [f"d{index}" for index in range(len(grades))],
        }
    )
    judgments = pd.DataFrame(
        {
            "query": ["desk"] * len(grades) + ["unmatched"],
            "doc_id": [*corpus["doc_id"], "not-in-corpus"],
            "relevance": [*grades, 0.25],
        }
    )

    samples = sample_judgments(
        corpus,
        judgments,
        grade_column="relevance",
        sample_size=4,
        seed=42,
    )

    assert set(samples) == {"desk", "unmatched"}
    assert samples["unmatched"] == []
    assert len(samples["desk"]) == 4
    assert {grade for _, grade in samples["desk"]} == {0.0, 0.1, 0.5, 1.0}
    assert all(isinstance(index, int) for index, _ in samples["desk"])
