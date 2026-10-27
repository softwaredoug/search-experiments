import pandas as pd
import pytest

from scripts.per_query_ecom.analyze_gpt6_luna_esci_study import (
    generate_analysis,
    write_markdown_report,
)


STRATEGY_NAME = "agentic_gpt6_luna_bm25_e5"
CACHE_SALTS = tuple(f"round-{number}" for number in range(1, 10))


def _write_study_runs(tmp_path, score_runs):
    summary_rows = []
    corpus_rows = [
        {"doc_id": "ideal-1", "title": "Ideal second", "description": "Ideal desc 1"},
        {"doc_id": "ideal-2", "title": "Ideal first", "description": "Ideal desc 2"},
    ]
    for run_index, cache_salt in enumerate(CACHE_SALTS):
        reports_path = tmp_path / f"reports-{run_index}"
        reports_path.mkdir()
        per_query = [
            {
                "query_id": query_id,
                "query": query,
                "ndcg_g": score_runs[query][run_index],
            }
            for query_id, query in (
                (1, "query one"),
                (2, "query two"),
                (3, "non-English"),
            )
        ]
        pd.DataFrame(per_query).to_csv(reports_path / "per_query.csv", index=False)

        result_rows = []
        for query_id, query in (
            (1, "query one"),
            (2, "query two"),
            (3, "non-English"),
        ):
            doc_id = f"actual-{query_id}-{run_index}"
            result_rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "doc_id": doc_id,
                    "rank": 1,
                    "grade": None if query_id == 1 else 0,
                }
            )
            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "title": f"Run {run_index} result for {query}",
                    "description": f"Full description from run {run_index}",
                }
            )
        pd.DataFrame(result_rows).to_parquet(
            reports_path / "full_results.parquet", index=False
        )
        summary_rows.append(
            {
                "dataset": "esci",
                "strategy_name": STRATEGY_NAME,
                "cache_salt": cache_salt,
                "run_id": f"run-{run_index}",
                "reports_path": str(reports_path),
            }
        )

    input_path = tmp_path / "summary.csv"
    pd.DataFrame(summary_rows).to_csv(input_path, index=False)
    judgments = pd.DataFrame(
        {
            "query_id": [1, 1, 2, 3],
            "query": ["query one", "query one", "query two", "non-English"],
            "doc_id": ["ideal-1", "ideal-2", "ideal-1", "ideal-1"],
            "grade": [2, 3, 1, 1],
            "product_locale": ["us", "us", "us", "jp"],
        }
    )
    corpus = pd.DataFrame(corpus_rows)
    return input_path, judgments, corpus


def test_quartile_selection_is_inclusive_and_requires_n_of_n_runs(tmp_path):
    input_path, judgments, corpus = _write_study_runs(
        tmp_path,
        {
            "query one": [0.25] * 9,
            "query two": [0.1] * 8 + [0.3],
            "non-English": [0.0] * 9,
        },
    )

    selected_path, paired_path = generate_analysis(
        input_path=input_path,
        output_dir=tmp_path / "quartile-analysis",
        judgments=judgments,
        corpus=corpus,
        by_quartile=0.25,
        by_quartile_num_rounds=9,
    )

    selected = pd.read_csv(selected_path)
    assert selected["query"].tolist() == ["query one"]
    assert selected["quartile_runs"].tolist() == [9]
    assert selected["scored_runs"].tolist() == [9]
    assert selected["mean_ndcg_g"].tolist() == pytest.approx([0.25])

    paired = pd.read_parquet(paired_path)
    assert len(paired) == 18
    assert paired["ideal_doc_id"].tolist() == ["ideal-2", "ideal-1"] * 9
    assert paired["ideal_grade"].tolist() == [3, 2] * 9
    assert paired["ideal_title"].tolist() == ["Ideal first", "Ideal second"] * 9
    assert paired["ideal_description"].tolist() == ["Ideal desc 2", "Ideal desc 1"] * 9
    assert paired["rhs_cache_salt"].tolist() == [salt for salt in CACHE_SALTS for _ in range(2)]
    assert paired["rhs_title"].iloc[0] == "Run 0 result for query one"
    assert paired["rhs_description"].iloc[0] == "Full description from run 0"


def test_variance_selection_exports_ideal_worst_and_best_runs(tmp_path):
    input_path, judgments, corpus = _write_study_runs(
        tmp_path,
        {
            "query one": [0.0, 0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4],
            "query two": [0.3, 0.5, 0.35, 0.45, 0.4, 0.4, 0.38, 0.42, 0.4],
            "non-English": [0.4] * 9,
        },
    )

    selected_path, paired_path = generate_analysis(
        input_path=input_path,
        output_dir=tmp_path / "variance-analysis",
        judgments=judgments,
        corpus=corpus,
        by_variance=1,
    )

    selected = pd.read_csv(selected_path)
    assert selected["query"].tolist() == ["query one"]
    assert selected["stddev_ndcg_g"].iloc[0] == pytest.approx(
        pd.Series([0.0, 0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4]).std(ddof=0)
    )

    paired = pd.read_parquet(paired_path)
    assert paired["worst_cache_salt"].unique().tolist() == ["round-1"]
    assert paired["worst_ndcg_g"].unique().tolist() == [0.0]
    assert paired["worst_doc_id"].iloc[0] == "actual-1-0"
    assert paired["best_cache_salt"].unique().tolist() == ["round-2"]
    assert paired["best_ndcg_g"].unique().tolist() == [0.8]
    assert paired["best_doc_id"].iloc[0] == "actual-1-1"
    assert paired["ideal_doc_id"].tolist() == ["ideal-2", "ideal-1"]


def test_markdown_report_contains_both_populations_and_hover_descriptions(tmp_path):
    input_path, judgments, corpus = _write_study_runs(
        tmp_path,
        {
            "query one": [0.1] * 9,
            "query two": [0.0, 0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5, 0.4],
            "non-English": [0.4] * 9,
        },
    )
    output_dir = tmp_path / "analysis"
    generate_analysis(
        input_path=input_path,
        output_dir=output_dir,
        judgments=judgments,
        corpus=corpus,
        by_quartile=0.25,
        by_quartile_num_rounds=9,
    )
    generate_analysis(
        input_path=input_path,
        output_dir=output_dir,
        judgments=judgments,
        corpus=corpus,
        by_variance=1,
    )

    report_path = write_markdown_report(
        input_path=input_path,
        output_dir=output_dir,
        report_path=tmp_path / "per_query_esci.md",
    )
    report = report_path.read_text(encoding="utf-8")

    assert "## Confidently Wrong" in report
    assert "## All over the place" in report
    assert "Ideal rank" in report
    assert "Worst run rank" in report
    assert "Best run rank" in report
    assert 'title="Full description from run 0"' in report
    assert "<img" not in report
    assert "en-NDCG" not in report
