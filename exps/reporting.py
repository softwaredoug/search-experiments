import csv
import json
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReportsWritten:
    run_id: str
    reports_path: Path


def _current_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def _append_csv_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_rows = []
    fieldnames = None
    file_exists = output_path.exists()
    if file_exists:
        with output_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            existing_rows = list(reader)

    if not fieldnames:
        fieldnames = list(rows[0])
    else:
        new_fields = list(
            dict.fromkeys(key for row in rows for key in row if key not in fieldnames)
        )
        if new_fields:
            fieldnames.extend(new_fields)
            with output_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(existing_rows)

    write_header = not file_exists or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _summary_row(
    result,
    *,
    dataset: str,
    seed: int,
    cache_salt: str | None,
    run_id: str,
    commit_sha: str,
    reports_path: Path,
) -> dict:
    return {
        "dataset": dataset,
        "run_id": run_id,
        "commit_sha": commit_sha,
        "seed": seed,
        "cache_salt": cache_salt or "",
        "strategy_name": result.strategy_name,
        "strategy_params": json.dumps(result.strategy_params, sort_keys=True),
        "metric_name": result.metric_name or "",
        "reports_path": str(reports_path),
        **(result.summary or {}),
    }


def _per_query_rows(
    result,
    *,
    dataset: str,
    seed: int,
    cache_salt: str | None,
    run_id: str,
    commit_sha: str,
    reports_path: Path,
) -> list[dict]:
    graded = result.graded
    if graded is None or graded.empty:
        raise ValueError("Per-query CSV output requires graded benchmark results.")
    query_keys = ["query_id", "query"]
    missing_query_keys = set(query_keys).difference(graded.columns)
    if missing_query_keys:
        raise ValueError(
            "Per-query results are missing query identifiers: "
            + ", ".join(sorted(missing_query_keys))
        )

    metric_columns = [
        column
        for column in (
            "ndcg_m",
            "ndcg_g",
            "max_ndcg_m",
            "mrr",
            "agreement",
            "pairwise_agreement",
        )
        if column in graded.columns
    ]
    resource_columns = sorted(
        column for column in graded.columns if column.startswith("resource.")
    )
    value_columns = [*metric_columns, *resource_columns]
    grouped = graded.groupby(query_keys, sort=False, dropna=False)
    for column in value_columns:
        if grouped[column].nunique(dropna=True).gt(1).any():
            raise ValueError(f"Per-query field {column!r} varies within a query.")
    per_query = grouped[value_columns].first().reset_index()

    metadata = {
        "run_id": run_id,
        "dataset": dataset,
        "commit_sha": commit_sha,
        "seed": seed,
        "cache_salt": cache_salt or "",
        "strategy_name": result.strategy_name,
        "metric_name": result.metric_name or "",
        "reports_path": str(reports_path),
    }
    rows = []
    for query_row in per_query.to_dict("records"):
        row = {**metadata}
        for key in query_keys:
            row[key] = query_row[key]
        for column in value_columns:
            value = query_row[column]
            row[column] = "" if pd.isna(value) else value
        rows.append(row)
    return rows


def _full_results_frame(
    result,
    *,
    dataset: str,
    seed: int,
    cache_salt: str | None,
    run_id: str,
    commit_sha: str,
) -> pd.DataFrame:
    graded = result.graded
    if graded is None or graded.empty:
        raise ValueError("Full-results Parquet output requires graded benchmark results.")
    required_columns = {"query_id", "query", "doc_id", "rank", "grade", "judged"}
    missing_columns = required_columns.difference(graded.columns)
    if missing_columns:
        raise ValueError(
            "Full results are missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    output = graded.loc[graded["rank"] <= 10].copy()
    output["doc_id"] = output["doc_id"].astype("string")
    output["judged"] = output["judged"].astype(bool)
    output["grade"] = pd.to_numeric(output["grade"], errors="raise").astype("Float64")
    output.loc[~output["judged"], "grade"] = pd.NA
    output["dataset"] = dataset
    output["run_id"] = run_id
    output["commit_sha"] = commit_sha
    output["seed"] = seed
    output["cache_salt"] = cache_salt or ""
    output["strategy_name"] = result.strategy_name
    output["strategy_params"] = json.dumps(result.strategy_params, sort_keys=True)
    output["metric_name"] = result.metric_name or ""
    return output


def write_benchmark_reports(
    result,
    *,
    dataset: str,
    seed: int,
    artifact_path: str | Path | None,
    cache_salt: str | None = None,
    summary_csv: str | Path | None = None,
) -> ReportsWritten:
    """Write standard reports for a full benchmark run and return their location."""
    run_id = uuid.uuid4().hex
    artifact_root = (
        Path(artifact_path).expanduser()
        if artifact_path is not None
        else Path(tempfile.mkdtemp(prefix="search-experiments-run-"))
    )
    artifact_root = artifact_root.resolve()
    reports_path = artifact_root / "reports" / run_id
    commit_sha = _current_commit_sha()
    per_query_rows = _per_query_rows(
        result,
        dataset=dataset,
        seed=seed,
        cache_salt=cache_salt,
        run_id=run_id,
        commit_sha=commit_sha,
        reports_path=reports_path,
    )
    full_results = _full_results_frame(
        result,
        dataset=dataset,
        seed=seed,
        cache_salt=cache_salt,
        run_id=run_id,
        commit_sha=commit_sha,
    )
    reports_path.mkdir(parents=True, exist_ok=False)
    _append_csv_rows(reports_path / "per_query.csv", per_query_rows)
    full_results.to_parquet(
        reports_path / "full_results.parquet", index=False, compression="zstd"
    )
    if summary_csv is not None:
        _append_csv_rows(
            summary_csv,
            [
                _summary_row(
                    result,
                    dataset=dataset,
                    seed=seed,
                    cache_salt=cache_salt,
                    run_id=run_id,
                    commit_sha=commit_sha,
                    reports_path=reports_path,
                )
            ],
        )

    return ReportsWritten(run_id=run_id, reports_path=reports_path)
