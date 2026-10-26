#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "research/results/gpt6_luna_esci_bm25_e5_study.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "research/results/gpt6_luna_esci_bm25_e5_study_analysis"


def _string_id(value) -> str:
    return "" if pd.isna(value) else str(value)


def _read_run_rows(
    input_path: Path,
    *,
    strategies: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    summary = pd.read_csv(input_path)
    required = {"dataset", "strategy_name", "cache_salt", "run_id", "reports_path"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(
            f"{input_path} is missing required columns: {', '.join(sorted(missing))}."
        )

    selected = summary.loc[summary["dataset"].eq("esci")].copy()
    selected["strategy_name"] = selected["strategy_name"].fillna("").astype(str)
    selected["cache_salt"] = selected["cache_salt"].fillna("").astype(str)
    if strategies:
        selected = selected.loc[selected["strategy_name"].isin(strategies)]
    if selected.empty:
        raise ValueError("No ESCI runs matched the requested strategy filters.")
    if selected["run_id"].duplicated().any():
        raise ValueError("The selected summary rows contain duplicate run IDs.")

    run_rows = []
    for row in selected.to_dict("records"):
        reports_path = str(row["reports_path"])
        if not reports_path or reports_path == "nan":
            raise ValueError(f"Run {row['run_id']!r} has no reports_path.")
        run = {
            "strategy_name": row["strategy_name"],
            "cache_salt": row["cache_salt"],
            "run_id": _string_id(row["run_id"]),
            "reports_path": reports_path,
        }
        run["score_column"] = (
            f"ndcg_g__{run['strategy_name']}__{run['cache_salt']}"
        )
        run_rows.append(run)
    score_columns = [run["score_column"] for run in run_rows]
    if len(set(score_columns)) != len(score_columns):
        raise ValueError("Each strategy/cache-salt pair must identify one run.")
    return run_rows


def _load_run_data(
    run_rows: list[dict[str, str]],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    score_frames = []
    results_by_run = {}
    for run in run_rows:
        reports_path = Path(run["reports_path"]).expanduser()
        per_query_path = reports_path / "per_query.csv"
        full_results_path = reports_path / "full_results.parquet"
        if not per_query_path.is_file() or not full_results_path.is_file():
            raise FileNotFoundError(
                f"Run {run['run_id']!r} is missing per_query.csv or "
                f"full_results.parquet under {reports_path}."
            )

        per_query = pd.read_csv(per_query_path)
        required = {"query_id", "query", "ndcg_g"}
        missing = required.difference(per_query.columns)
        if missing:
            raise ValueError(
                f"{per_query_path} is missing required columns: "
                f"{', '.join(sorted(missing))}."
            )
        per_query = per_query[["query_id", "query", "ndcg_g"]].copy()
        per_query["query_id"] = per_query["query_id"].map(_string_id)
        per_query["ndcg_g"] = pd.to_numeric(per_query["ndcg_g"], errors="raise")
        if per_query.duplicated(["query_id", "query"]).any():
            raise ValueError(f"{per_query_path} contains duplicate query rows.")
        per_query = per_query.rename(
            columns={"ndcg_g": run["score_column"]}
        )
        score_frames.append(per_query)

        results = pd.read_parquet(full_results_path)
        required_results = {"query_id", "query", "doc_id", "rank", "grade"}
        missing = required_results.difference(results.columns)
        if missing:
            raise ValueError(
                f"{full_results_path} is missing required columns: "
                f"{', '.join(sorted(missing))}."
            )
        results["query_id"] = results["query_id"].map(_string_id)
        results["doc_id"] = results["doc_id"].map(_string_id)
        results_by_run[run["run_id"]] = results

    scores = score_frames[0]
    for frame in score_frames[1:]:
        scores = scores.merge(
            frame, on=["query_id", "query"], how="outer", validate="one_to_one"
        )
    return scores, results_by_run


def _positive_query_keys(judgments: pd.DataFrame) -> set[tuple[str, str]]:
    relevant_grades = pd.to_numeric(judgments["grade"], errors="raise")
    return {
        (_string_id(row.query_id), row.query)
        for row in judgments.loc[relevant_grades.gt(0), ["query_id", "query"]]
        .drop_duplicates()
        .itertuples(index=False)
    }


def _select_queries(
    scores: pd.DataFrame,
    *,
    run_rows: list[dict[str, str]],
    judgments: pd.DataFrame,
    by_quartile: float | None,
    by_quartile_num_rounds: int,
    by_variance: int | None,
) -> pd.DataFrame:
    score_columns = [run["score_column"] for run in run_rows]
    output = scores.copy()
    positive_keys = _positive_query_keys(judgments)
    has_positive_judgments = pd.Series(
        [
            (query_id, query) in positive_keys
            for query_id, query in zip(output["query_id"], output["query"])
        ],
        index=output.index,
    )

    if by_quartile is not None:
        if not 0 <= by_quartile <= 1:
            raise ValueError("--by-quartile must be between 0 and 1.")
        selected_by_strategy = []
        strategy_names = dict.fromkeys(run["strategy_name"] for run in run_rows)
        for strategy_name in strategy_names:
            strategy_runs = [
                run for run in run_rows if run["strategy_name"] == strategy_name
            ]
            strategy_columns = [run["score_column"] for run in strategy_runs]
            if not 1 <= by_quartile_num_rounds <= len(strategy_runs):
                raise ValueError(
                    "--by-quartile-num-rounds must be between 1 and the selected "
                    f"run count for strategy {strategy_name!r}."
                )
            values = output[strategy_columns]
            candidate = output.copy()
            candidate["scored_runs"] = values.notna().sum(axis=1)
            candidate["mean_ndcg_g"] = values.mean(axis=1)
            candidate["stddev_ndcg_g"] = values.std(axis=1, ddof=0)
            candidate["quartile_runs"] = values.le(by_quartile).sum(axis=1)
            complete = (
                candidate["scored_runs"].eq(len(strategy_runs))
                & has_positive_judgments
            )
            selected = candidate.loc[
                complete
                & candidate["quartile_runs"].ge(by_quartile_num_rounds)
            ].copy()
            selected["selection_strategy"] = strategy_name
            selected_by_strategy.append(selected)
        if not selected_by_strategy:
            return output.iloc[0:0].assign(
                scored_runs=pd.Series(dtype=int),
                mean_ndcg_g=pd.Series(dtype=float),
                stddev_ndcg_g=pd.Series(dtype=float),
                quartile_runs=pd.Series(dtype=int),
                selection_strategy=pd.Series(dtype=str),
            )
        return pd.concat(selected_by_strategy, ignore_index=True).sort_values(
            ["mean_ndcg_g", "query_id", "selection_strategy"], kind="stable"
        ).reset_index(drop=True)

    if by_variance is None or by_variance < 1:
        raise ValueError("--by-variance must be a positive query count.")
    values = output[score_columns]
    output["scored_runs"] = values.notna().sum(axis=1)
    output["mean_ndcg_g"] = values.mean(axis=1)
    output["stddev_ndcg_g"] = values.std(axis=1, ddof=0)
    output["quartile_runs"] = 0
    complete = output["scored_runs"].eq(len(run_rows)) & has_positive_judgments
    return (
        output.loc[complete]
        .sort_values(
            ["stddev_ndcg_g", "query_id"], ascending=[False, True], kind="stable"
        )
        .head(by_variance)
        .reset_index(drop=True)
    )


def _text_columns(corpus: pd.DataFrame) -> pd.DataFrame:
    if "doc_id" not in corpus:
        raise ValueError("ESCI corpus is missing the doc_id column.")
    docs = corpus.copy()
    docs["doc_id"] = docs["doc_id"].map(_string_id)
    for column in ("title", "description"):
        if column not in docs:
            docs[column] = ""
        docs[column] = docs[column].fillna("").astype(str)
    return docs[["doc_id", "title", "description"]].drop_duplicates("doc_id")


def _ideal_results_by_query(
    judgments: pd.DataFrame,
    corpus: pd.DataFrame,
    *,
    top_k: int,
) -> dict[tuple[str, str], list[dict]]:
    required = {"query_id", "query", "doc_id", "grade"}
    missing = required.difference(judgments.columns)
    if missing:
        raise ValueError(
            f"ESCI judgments are missing columns: {', '.join(sorted(missing))}."
        )
    judged = judgments[["query_id", "query", "doc_id", "grade"]].copy()
    judged["query_id"] = judged["query_id"].map(_string_id)
    judged["doc_id"] = judged["doc_id"].map(_string_id)
    judged["grade"] = pd.to_numeric(judged["grade"], errors="raise")
    judged = (
        judged.sort_values(
            ["query_id", "query", "grade", "doc_id"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .drop_duplicates(["query_id", "query", "doc_id"], keep="first")
    )
    judged = judged.loc[judged["grade"].gt(0)].merge(
        _text_columns(corpus), on="doc_id", how="left", validate="many_to_one"
    )
    judged[["title", "description"]] = judged[["title", "description"]].fillna("")

    ideal: dict[tuple[str, str], list[dict]] = {}
    for (query_id, query), group in judged.groupby(
        ["query_id", "query"], sort=False
    ):
        ideal[(query_id, query)] = [
            {
                "rank": rank,
                "doc_id": row.doc_id,
                "grade": row.grade,
                "title": row.title,
                "description": row.description,
            }
            for rank, row in enumerate(group.head(top_k).itertuples(index=False), start=1)
        ]
    return ideal


def _run_documents(
    run: dict[str, str],
    *,
    query_id: str,
    query: str,
    results_by_run: dict[str, pd.DataFrame],
    docs: pd.DataFrame,
    top_k: int,
) -> list[dict]:
    results = results_by_run[run["run_id"]].copy()
    results["rank"] = pd.to_numeric(results["rank"], errors="raise")
    results["grade"] = pd.to_numeric(results["grade"], errors="coerce")
    results = results.loc[
        results["query_id"].eq(query_id) & results["query"].eq(query)
    ].sort_values("rank", kind="stable").head(top_k)
    results = results.merge(
        docs.rename(columns={"title": "variant_title", "description": "variant_description"}),
        on="doc_id",
        how="left",
        validate="many_to_one",
    )
    return [
        {
            "rank": row.rank,
            "doc_id": row.doc_id,
            "grade": row.grade,
            "title": _string_id(row.variant_title),
            "description": _string_id(row.variant_description),
        }
        for row in results.itertuples(index=False)
    ]


def _variant_metadata(run: dict[str, str], ndcg_g: float) -> dict[str, object]:
    return {
        "strategy_name": run["strategy_name"],
        "cache_salt": run["cache_salt"],
        "run_id": run["run_id"],
        "ndcg_g": ndcg_g,
    }


def _side_by_side_rows(
    selected_queries: pd.DataFrame,
    *,
    mode: str,
    run_rows: list[dict[str, str]],
    scores_long: pd.DataFrame,
    results_by_run: dict[str, pd.DataFrame],
    judgments: pd.DataFrame,
    corpus: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    if selected_queries.empty:
        return pd.DataFrame()
    ideal_by_query = _ideal_results_by_query(judgments, corpus, top_k=top_k)
    docs = _text_columns(corpus)
    score_columns = [run["score_column"] for run in run_rows]
    rows = []

    for query_data in selected_queries.to_dict("records"):
        query_id = query_data["query_id"]
        query = query_data["query"]
        query_scores = scores_long.loc[
            scores_long["query_id"].eq(query_id)
            & scores_long["query"].eq(query)
        ].dropna(subset=["ndcg_g"])
        if mode == "quartile":
            selected_strategy = query_data["selection_strategy"]
            strategy_scores = query_scores.loc[
                query_scores["strategy_name"].eq(selected_strategy)
            ]
            by_id = {run["run_id"]: run for run in run_rows}
            variant_runs = [
                ("rhs", by_id[row.run_id], float(row.ndcg_g))
                for row in strategy_scores.itertuples(index=False)
            ]
        else:
            worst = query_scores.sort_values("ndcg_g", kind="stable").iloc[0]
            best = query_scores.sort_values("ndcg_g", kind="stable").iloc[-1]
            by_id = {run["run_id"]: run for run in run_rows}
            variant_runs = [
                ("worst", by_id[worst["run_id"]], float(worst["ndcg_g"])),
                ("best", by_id[best["run_id"]], float(best["ndcg_g"])),
            ]

        ideal_rows = ideal_by_query.get((query_id, query), [])
        variant_docs = [
            (
                label,
                run,
                ndcg_g,
                _run_documents(
                    run,
                    query_id=query_id,
                    query=query,
                    results_by_run=results_by_run,
                    docs=docs,
                    top_k=top_k,
                ),
            )
            for label, run, ndcg_g in variant_runs
        ]

        def append_comparison_rows(variants):
            row_count = max(
                [len(ideal_rows), *(len(documents) for _, _, _, documents in variants)]
            )
            for rank_index in range(row_count):
                ideal = ideal_rows[rank_index] if rank_index < len(ideal_rows) else {}
                row: dict[str, object] = {
                    "dataset": "esci",
                    "query_id": query_id,
                    "query": query,
                    "mean_ndcg_g": query_data["mean_ndcg_g"],
                    "stddev_ndcg_g": query_data["stddev_ndcg_g"],
                    "scored_runs": query_data["scored_runs"],
                    "quartile_runs": query_data["quartile_runs"],
                    "selection_strategy": query_data.get("selection_strategy", ""),
                    **{column: query_data[column] for column in score_columns},
                    "ideal_rank": ideal.get("rank"),
                    "ideal_doc_id": ideal.get("doc_id"),
                    "ideal_grade": ideal.get("grade"),
                    "ideal_title": ideal.get("title"),
                    "ideal_description": ideal.get("description"),
                }
                for label, run, ndcg_g, documents in variants:
                    actual = (
                        documents[rank_index]
                        if rank_index < len(documents)
                        else {}
                    )
                    metadata = _variant_metadata(run, ndcg_g)
                    row.update(
                        {f"{label}_{key}": value for key, value in metadata.items()}
                    )
                    row.update(
                        {
                            f"{label}_{key}": actual.get(key)
                            for key in (
                                "rank",
                                "doc_id",
                                "grade",
                                "title",
                                "description",
                            )
                        }
                    )
                rows.append(row)

        if mode == "quartile":
            for variant in variant_docs:
                append_comparison_rows([variant])
        else:
            append_comparison_rows(variant_docs)
    return pd.DataFrame(rows)


def generate_analysis(
    *,
    input_path: Path,
    output_dir: Path,
    judgments: pd.DataFrame,
    corpus: pd.DataFrame,
    by_quartile: float | None = None,
    by_quartile_num_rounds: int = 9,
    by_variance: int | None = None,
    strategies: tuple[str, ...] = (),
    top_k: int = 10,
) -> tuple[Path, Path]:
    if (by_quartile is None) == (by_variance is None):
        raise ValueError("Choose exactly one of by_quartile or by_variance.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    run_rows = _read_run_rows(input_path, strategies=strategies)
    if by_quartile is not None:
        if not 0 <= by_quartile <= 1:
            raise ValueError("by_quartile must be between 0 and 1.")
        mode = "quartile"
    else:
        if by_variance is None or by_variance < 1:
            raise ValueError("by_variance must be a positive query count.")
        mode = "variance"

    scores_wide, results_by_run = _load_run_data(run_rows)
    score_columns = [run["score_column"] for run in run_rows]
    selected_queries = _select_queries(
        scores_wide,
        run_rows=run_rows,
        judgments=judgments,
        by_quartile=by_quartile,
        by_quartile_num_rounds=by_quartile_num_rounds,
        by_variance=by_variance,
    )
    scores_long = scores_wide.melt(
        id_vars=["query_id", "query"],
        value_vars=score_columns,
        var_name="score_column",
        value_name="ndcg_g",
    ).merge(
        pd.DataFrame(
            [
                {
                    "score_column": run["score_column"],
                    "strategy_name": run["strategy_name"],
                    "cache_salt": run["cache_salt"],
                    "run_id": run["run_id"],
                }
                for run in run_rows
            ]
        ),
        on="score_column",
        how="left",
        validate="many_to_one",
    )
    paired = _side_by_side_rows(
        selected_queries,
        mode=mode,
        run_rows=run_rows,
        scores_long=scores_long,
        results_by_run=results_by_run,
        judgments=judgments,
        corpus=corpus,
        top_k=top_k,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / f"queries_by_{mode}.csv"
    paired_path = output_dir / f"query_results_by_{mode}.parquet"
    selected_queries.to_csv(selected_path, index=False)
    paired.to_parquet(paired_path, index=False, compression="zstd")
    return selected_path, paired_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select ESCI queries by NDCG-G threshold or run variance and export "
            "judgment-ideal vs. run-result rows."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--by-quartile",
        type=float,
        help="Select queries with NDCG-G at or below this threshold in enough runs.",
    )
    selection.add_argument(
        "--by-variance",
        type=int,
        help="Select the top N queries by NDCG-G standard deviation across runs.",
    )
    parser.add_argument("--by-quartile-num-rounds", type=int, default=9)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Optional strategy names; defaults to every strategy in the ESCI input.",
    )
    args = parser.parse_args()

    from exps.datasets import get_dataset

    dataset = get_dataset("esci", ensure_snowball=False)
    selected_path, paired_path = generate_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        judgments=dataset.judgments,
        corpus=dataset.corpus,
        by_quartile=args.by_quartile,
        by_quartile_num_rounds=args.by_quartile_num_rounds,
        by_variance=args.by_variance,
        strategies=tuple(args.strategies or ()),
        top_k=args.top_k,
    )
    print(f"Wrote {selected_path}")
    print(f"Wrote {paired_path}")


if __name__ == "__main__":
    main()
