#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
from pathlib import Path
from numbers import Real

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT_DIR / "research/results/gpt6_luna_esci_bm25_e5_study.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "research/results/gpt6_luna_esci_bm25_e5_study_analysis"
ENGLISH_LOCALE_ALIASES = {
    "en", "eng", "english", "en-us", "en-gb", "us", "uk", "gb"
}


def _string_id(value) -> str:
    return "" if pd.isna(value) else str(value)


def _filter_judgments_by_language(
    judgments: pd.DataFrame, language: str
) -> pd.DataFrame:
    """Filter ESCI judgments by language; ESCI's US locale is English."""
    language = language.strip().casefold()
    locale_column = next(
        (
            column
            for column in (
                "query_language",
                "language",
                "query_locale",
                "locale",
                "product_locale",
            )
            if column in judgments.columns
        ),
        None,
    )
    if locale_column is None:
        raise ValueError("ESCI judgments have no language or locale column.")
    accepted_values = (
        ENGLISH_LOCALE_ALIASES if language == "en" else {language}
    )
    values = (
        judgments[locale_column]
        .astype("string")
        .str.strip()
        .str.casefold()
        .str.replace("_", "-", regex=False)
    )
    filtered = judgments.loc[values.isin(accepted_values)].copy()
    if filtered.empty:
        raise ValueError(f"No ESCI judgments matched language {language!r}.")
    return filtered


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


def _selected_doc_ids(
    selected_queries: pd.DataFrame,
    *,
    judgments: pd.DataFrame,
    results_by_run: dict[str, pd.DataFrame],
    top_k: int,
) -> set[str]:
    query_keys = set(zip(selected_queries["query_id"], selected_queries["query"]))
    selected_judgments = judgments.loc[
        [
            (_string_id(query_id), query) in query_keys
            for query_id, query in zip(judgments["query_id"], judgments["query"])
        ]
    ].copy()
    if not selected_judgments.empty:
        selected_judgments["grade"] = pd.to_numeric(
            selected_judgments["grade"], errors="raise"
        )
        selected_judgments["doc_id"] = selected_judgments["doc_id"].map(_string_id)
        ideal_docs = (
            selected_judgments.loc[selected_judgments["grade"].gt(0)]
            .sort_values(
                ["query_id", "query", "grade", "doc_id"],
                ascending=[True, True, False, True],
                kind="stable",
            )
            .drop_duplicates(["query_id", "query", "doc_id"])
            .groupby(["query_id", "query"], sort=False)
            .head(top_k)
        )
        doc_ids = set(ideal_docs["doc_id"])
    else:
        doc_ids = set()

    for results in results_by_run.values():
        selected_results = results.loc[
            [
                (_string_id(query_id), query) in query_keys
                for query_id, query in zip(results["query_id"], results["query"])
            ]
        ]
        doc_ids.update(selected_results["doc_id"].map(_string_id))
    return doc_ids


def _load_product_texts(doc_ids: set[str]) -> pd.DataFrame:
    """Load only the selected product text, avoiding ESCI's full search index build."""
    if not doc_ids:
        return pd.DataFrame(columns=["doc_id", "title", "description"])

    from cheat_at_search.esci_data import fetch_esci

    products_path = (
        fetch_esci()
        / "shopping_queries_dataset"
        / "shopping_queries_dataset_products.parquet"
    )
    products = pd.read_parquet(
        products_path,
        columns=[
            "product_id",
            "product_locale",
            "product_title",
            "product_description",
        ],
        filters=[("product_id", "in", sorted(doc_ids))],
    )
    products = products.rename(
        columns={
            "product_id": "doc_id",
            "product_title": "title",
            "product_description": "description",
        }
    )
    products["doc_id"] = products["doc_id"].map(_string_id)
    products["title"] = products["title"].fillna("").astype(str)

    from lxml import html
    from lxml.etree import ParserError

    def clean_description(value) -> str:
        if pd.isna(value) or not str(value).strip():
            return ""
        try:
            return " ".join(html.fromstring(value).text_content().split())
        except (ParserError, ValueError):
            return " ".join(str(value).split())

    products["description"] = products["description"].map(clean_description)
    products["_locale_priority"] = products["product_locale"].eq("us").astype(int)
    products = products.sort_values("_locale_priority", ascending=False, kind="stable")
    products = products.drop_duplicates("doc_id", keep="first")
    loaded_ids = set(products["doc_id"])
    missing = doc_ids.difference(loaded_ids)
    if missing:
        products = pd.concat(
            [
                products,
                pd.DataFrame(
                    {
                        "doc_id": sorted(missing),
                        "title": "",
                        "description": "",
                    }
                ),
            ],
            ignore_index=True,
        )
    return products[["doc_id", "title", "description"]]


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
    corpus: pd.DataFrame | None = None,
    by_quartile: float | None = None,
    by_quartile_num_rounds: int = 9,
    by_variance: int | None = None,
    query_language: str = "en",
    strategies: tuple[str, ...] = (),
    top_k: int = 10,
) -> tuple[Path, Path]:
    if (by_quartile is None) == (by_variance is None):
        raise ValueError("Choose exactly one of by_quartile or by_variance.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    run_rows = _read_run_rows(input_path, strategies=strategies)
    judgments = _filter_judgments_by_language(judgments, query_language)
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
    if corpus is None:
        corpus = _load_product_texts(
            _selected_doc_ids(
                selected_queries,
                judgments=judgments,
                results_by_run=results_by_run,
                top_k=top_k,
            )
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


def _markdown_cell(value) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, Real) and float(value).is_integer():
        value = int(value)
    return str(value).replace("|", "&#124;").replace("\n", " ").replace("\r", " ")


def _hover_title(title, description) -> str:
    title_text = "No title" if pd.isna(title) or not str(title).strip() else str(title)
    description_text = (
        "No product description available."
        if pd.isna(description) or not str(description).strip()
        else str(description)
    )
    visible = html.escape(title_text, quote=False).replace("|", "&#124;")
    tooltip = html.escape(" ".join(description_text.split()), quote=True).replace(
        "|", "&#124;"
    )
    return f'<span title="{tooltip}">{visible}</span>'


def _grade_label(value) -> str:
    if pd.isna(value):
        return "NA"
    grade = int(value)
    labels = {3: "E", 2: "S", 1: "C", 0: "I"}
    return f"{grade} ({labels.get(grade, '?')})"


def _render_results_table(
    rows: pd.DataFrame,
    *,
    right_sides: tuple[str, ...],
) -> list[str]:
    headers = ["Ideal rank", "Grade", "Doc ID", "Title"]
    for side in right_sides:
        label = {"rhs": "Run", "worst": "Worst run", "best": "Best run"}[side]
        headers.extend(
            [f"{label} rank", "Grade", "Doc ID", "Title"]
        )
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---:"] + ["---"] * (len(headers) - 1)) + " |"]
    for row in rows.to_dict("records"):
        cells = [
            _markdown_cell(row.get("ideal_rank")),
            _grade_label(row.get("ideal_grade")),
            _markdown_cell(row.get("ideal_doc_id")),
            _hover_title(row.get("ideal_title"), row.get("ideal_description")),
        ]
        for side in right_sides:
            cells.extend(
                [
                    _markdown_cell(row.get(f"{side}_rank")),
                    _grade_label(row.get(f"{side}_grade")),
                    _markdown_cell(row.get(f"{side}_doc_id")),
                    _hover_title(
                        row.get(f"{side}_title"), row.get(f"{side}_description")
                    ),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _round_score_columns(frame: pd.DataFrame, strategy_name: str) -> list[str]:
    prefix = f"ndcg_g__{strategy_name}__"
    return [column for column in frame.columns if column.startswith(prefix)]


def write_markdown_report(
    *,
    input_path: Path,
    output_dir: Path,
    report_path: Path,
) -> Path:
    weak_queries_path = output_dir / "queries_by_quartile.csv"
    weak_results_path = output_dir / "query_results_by_quartile.parquet"
    variable_queries_path = output_dir / "queries_by_variance.csv"
    variable_results_path = output_dir / "query_results_by_variance.parquet"
    required_files = (
        weak_queries_path,
        weak_results_path,
        variable_queries_path,
        variable_results_path,
    )
    missing = [path for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Run both query-selection modes first; missing "
            + ", ".join(str(path) for path in missing)
        )

    summary = pd.read_csv(input_path)
    summary = summary.loc[
        summary["dataset"].eq("esci")
        & summary["strategy_name"].eq("agentic_gpt6_luna_bm25_e5")
    ]
    if summary.empty:
        raise ValueError("No GPT-6 Luna BM25+E5 ESCI runs found in the input summary.")
    rounds = summary["cache_salt"].astype(str).drop_duplicates().tolist()
    strategy_name = "agentic_gpt6_luna_bm25_e5"

    from exps.datasets import get_dataset

    english_judgments = _filter_judgments_by_language(
        get_dataset("esci", ensure_snowball=False).judgments, "en"
    )
    english_keys = {
        (_string_id(query_id), query)
        for query_id, query in english_judgments[["query_id", "query"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    run_samples = []
    for run in summary.to_dict("records"):
        per_query = pd.read_csv(Path(run["reports_path"]) / "per_query.csv")
        keys = {
            (_string_id(query_id), query)
            for query_id, query in per_query[["query_id", "query"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        english_rows = per_query.loc[
            [
                (_string_id(query_id), query) in english_keys
                for query_id, query in zip(per_query["query_id"], per_query["query"])
            ]
        ]
        run_samples.append(
            {
                "cache_salt": run["cache_salt"],
                "query_keys": keys,
                "sample_count": len(keys),
                "english_count": len(keys.intersection(english_keys)),
                "mean_ndcg_g": pd.to_numeric(
                    english_rows["ndcg_g"], errors="raise"
                ).mean(),
            }
        )
    same_query_sample = all(
        run["query_keys"] == run_samples[0]["query_keys"] for run in run_samples[1:]
    )

    weak_queries = pd.read_csv(weak_queries_path)
    weak_scores = _round_score_columns(weak_queries, strategy_name)
    strictly_weak = weak_queries[weak_scores].lt(0.25).all(axis=1)
    weak_queries = weak_queries.loc[strictly_weak].copy()
    weak_details = pd.read_parquet(weak_results_path)
    weak_keys = set(zip(weak_queries["query_id"].astype(str), weak_queries["query"]))
    weak_details = weak_details.loc[
        [
            (_string_id(query_id), query) in weak_keys
            for query_id, query in zip(weak_details["query_id"], weak_details["query"])
        ]
    ]
    representative_salt = rounds[0]
    representative_strategy = (
        weak_details["rhs_strategy_name"].iloc[0]
        if not weak_details.empty
        else strategy_name
    )
    weak_details = weak_details.loc[
        weak_details["rhs_cache_salt"].eq(representative_salt)
        & weak_details["rhs_strategy_name"].eq(representative_strategy)
    ]
    unjudged_round1 = int(weak_details["rhs_grade"].isna().sum())
    judged_round1 = int(weak_details["rhs_grade"].notna().sum())
    irrelevant_round1 = int(weak_details["rhs_grade"].eq(0).sum())
    judged_queries_round1 = int(
        weak_details.groupby(["query_id", "query"])["rhs_grade"]
        .apply(lambda grades: grades.notna().any())
        .sum()
    )

    variable_queries = pd.read_csv(variable_queries_path)
    variance_scores = _round_score_columns(variable_queries, strategy_name)
    variable_details = pd.read_parquet(variable_results_path)
    above_one = []
    run_by_salt = {
        str(row.cache_salt): row for row in summary.itertuples(index=False)
    }
    for query in variable_queries.to_dict("records"):
        for score_column in variance_scores:
            score = float(query[score_column])
            if score <= 1:
                continue
            cache_salt = score_column.rsplit("__", 1)[-1]
            run = run_by_salt[cache_salt]
            ranked = pd.read_parquet(
                Path(run.reports_path) / "full_results.parquet"
            )
            query_results = ranked.loc[
                ranked["query_id"].map(_string_id).eq(_string_id(query["query_id"]))
                & ranked["query"].eq(query["query"])
            ]
            duplicate_count = int(query_results["doc_id"].astype(str).duplicated().sum())
            above_one.append(
                {
                    "query": query["query"],
                    "cache_salt": cache_salt,
                    "ndcg_g": score,
                    "duplicate_count": duplicate_count,
                }
            )

    sample_count = run_samples[0]["sample_count"]
    english_count = run_samples[0]["english_count"]
    sample_consistency = (
        "The same query sample appears in all nine runs."
        if same_query_sample
        else "The sampled query sets differ between runs."
    )
    lines = [
        "# ESCI: GPT-6 Luna BM25 + E5 per-query analysis",
        "",
        "> This report was generated by an LLM.",
        "",
        "## Scope and selection",
        "",
        (
            f"The study ran `{strategy_name}` for {sample_count} queries per round with seed 42 "
            f"and nine cache-salt rounds ({', '.join(rounds)}). English-query analysis "
            f"uses ESCI `product_locale=us`, the dataset's US-English locale ({english_count} "
            f"of {sample_count} sampled queries). Query selection is post-hoc. "
            f"{sample_consistency} NDCG-G is computed from the ESCI graded judgments. "
            "Only queries with at least one positive-grade judgment are included."
        ),
        "",
        "| Round | Sampled queries | English queries | Mean English NDCG-G |",
        "|---|---:|---:|---:|",
    ]
    lines.extend(
        "| "
        + " | ".join(
            [
                _markdown_cell(run["cache_salt"]),
                str(run["sample_count"]),
                str(run["english_count"]),
                f"{float(run['mean_ndcg_g']):.3f}",
            ]
        )
        + " |"
        for run in run_samples
    )
    lines.extend(
        [
            "",
            (
                "Ideal results use positive-grade judgments sorted by grade descending "
                "and document ID ascending, up to rank 10; they are not padded with "
                "grade-0 documents. Hover over a title to view its full product description. "
                "No product images are included. `NA` means the result was not judged; "
                "grade 0 is an explicit irrelevance judgment. ESCI grades are E=3, S=2, C=1, I=0."
            ),
            "",
            "## Confidently Wrong",
            "",
            (
                f"Selected {len(weak_queries)} English queries with NDCG-G below 0.25 "
                "in all nine runs. The detailed tables compare the ideal judged ranking "
                f"with {representative_salt}, the first recorded replay. In that replay, "
                f"{unjudged_round1} of {len(weak_details)} top-10 result rows were unjudged, "
                f"{judged_round1} were judged, and {irrelevant_round1} had explicit grade 0. "
                f"Only {judged_queries_round1} of {len(weak_queries)} queries had a judged "
                "top-10 result; unjudged results are not proof of irrelevance."
            ),
            "",
        ]
    )

    round_headers = [column.removeprefix(f"ndcg_g__{strategy_name}__") for column in weak_scores]
    lines.append(
        "| Query ID | Query | "
        + " | ".join(round_headers)
        + " | Mean | SD |"
    )
    lines.append("|---:|---|" + "---:|" * (len(round_headers) + 2))
    for query in weak_queries.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(query["query_id"]),
                    _markdown_cell(query["query"]),
                    *[f"{float(query[column]):.3f}" for column in weak_scores],
                    f"{float(query['mean_ndcg_g']):.3f}",
                    f"{float(query['stddev_ndcg_g']):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")

    for query in weak_queries.to_dict("records"):
        query_rows = weak_details.loc[
            weak_details["query_id"].eq(str(query["query_id"]))
            & weak_details["query"].eq(query["query"])
        ].sort_values("ideal_rank", kind="stable")
        score_text = " · ".join(
            f"{round_name}: {float(query[column]):.3f}"
            for round_name, column in zip(round_headers, weak_scores)
        )
        lines.extend(
            [
                f"### {query['query_id']} — {_markdown_cell(query['query'])}",
                "",
                f"NDCG-G by round: {score_text}; mean: **{float(query['mean_ndcg_g']):.3f}**.",
                "",
                *_render_results_table(query_rows, right_sides=("rhs",)),
                "",
            ]
        )

    lines.extend(
        [
            "## All over the place",
            "",
            (
                f"Selected the {len(variable_queries)} English queries with the highest "
                "population standard deviation of NDCG-G across the nine runs. The "
                "tables compare ideal judgments with the lowest- and highest-scoring "
                "run for each query."
            ),
            "",
        ]
    )
    variance_headers = [
        column.removeprefix(f"ndcg_g__{strategy_name}__")
        for column in variance_scores
    ]
    lines.append(
        "| Query ID | Query | "
        + " | ".join(variance_headers)
        + " | Mean | SD |"
    )
    lines.append("|---:|---|" + "---:|" * (len(variance_headers) + 2))
    for query in variable_queries.to_dict("records"):
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(query["query_id"]),
                    _markdown_cell(query["query"]),
                    *[f"{float(query[column]):.3f}" for column in variance_scores],
                    f"{float(query['mean_ndcg_g']):.3f}",
                    f"{float(query['stddev_ndcg_g']):.3f}",
                ]
            )
            + " |"
        )
    lines.append("")
    if above_one:
        anomaly_text = "; ".join(
            f"{item['query']} ({item['cache_salt']}: {item['ndcg_g']:.3f}, "
            f"{item['duplicate_count']} duplicate result rows)"
            for item in above_one
        )
        lines.extend(
            [
                (
                    "Metric note: NDCG-G exceeded 1.0 in runs with repeated document IDs "
                    f"in the ranking: {anomaly_text}. These duplicate results inflate "
                    "DCG and should be considered when interpreting the variance cases."
                ),
                "",
            ]
        )

    for query in variable_queries.to_dict("records"):
        query_rows = variable_details.loc[
            variable_details["query_id"].eq(str(query["query_id"]))
            & variable_details["query"].eq(query["query"])
        ].sort_values("ideal_rank", kind="stable")
        score_text = " · ".join(
            f"{round_name}: {float(query[column]):.3f}"
            for round_name, column in zip(variance_headers, variance_scores)
        )
        worst_salt = query_rows["worst_cache_salt"].iloc[0]
        best_salt = query_rows["best_cache_salt"].iloc[0]
        lines.extend(
            [
                f"### {query['query_id']} — {_markdown_cell(query['query'])}",
                "",
                (
                    f"NDCG-G by round: {score_text}; mean: "
                    f"**{float(query['mean_ndcg_g']):.3f}**; "
                    f"SD: **{float(query['stddev_ndcg_g']):.3f}**. "
                    f"Worst: {worst_salt}; best: {best_salt}."
                ),
                "",
                *_render_results_table(query_rows, right_sides=("worst", "best")),
                "",
            ]
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select ESCI queries by NDCG-G threshold or run variance and export "
            "judgment-ideal vs. run-result rows."
        )
    )
    selection = parser.add_mutually_exclusive_group(required=False)
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
        "--query-language",
        default="en",
        help="Query language to analyze; ESCI's us product locale maps to en.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Optional strategy names; defaults to every strategy in the ESCI input.",
    )
    parser.add_argument(
        "--write-report",
        action="store_true",
        help="Render research/agentic/per_query_esci.md from both analysis outputs.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=ROOT_DIR / "research/agentic/per_query_esci.md",
    )
    args = parser.parse_args()

    if args.write_report:
        report_path = write_markdown_report(
            input_path=args.input,
            output_dir=args.output_dir,
            report_path=args.report_path,
        )
        print(f"Wrote {report_path}")
        return
    if args.by_quartile is None and args.by_variance is None:
        parser.error("Select --by-quartile, --by-variance, or --write-report.")

    from exps.datasets import get_dataset

    dataset = get_dataset("esci", ensure_snowball=False)
    selected_path, paired_path = generate_analysis(
        input_path=args.input,
        output_dir=args.output_dir,
        judgments=dataset.judgments,
        by_quartile=args.by_quartile,
        by_quartile_num_rounds=args.by_quartile_num_rounds,
        by_variance=args.by_variance,
        query_language=args.query_language,
        strategies=tuple(args.strategies or ()),
        top_k=args.top_k,
    )
    print(f"Wrote {selected_path}")
    print(f"Wrote {paired_path}")


if __name__ == "__main__":
    main()
