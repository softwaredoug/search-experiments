from __future__ import annotations

import os
import re
from pathlib import PurePosixPath
from time import perf_counter

import pandas as pd


def _slugify(value: str) -> str:
    lowered = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def _slugify_series(values: pd.Series) -> pd.Series:
    return (
        values.str.lower()
        .str.replace(r"[^a-z0-9]+", "-", regex=True)
        .str.replace(r"-+", "-", regex=True)
        .str.strip("-")
    )


def _normalize_dir(path: str) -> str:
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    trimmed = path.strip() or "/"
    if not trimmed.startswith("/"):
        trimmed = f"/{trimmed}"
    if trimmed != "/" and not trimmed.endswith("/"):
        trimmed = f"{trimmed}/"
    return trimmed


def _normalize_glob(prefix: str, glob: str) -> str:
    if not isinstance(glob, str):
        raise ValueError("glob must be a string")
    if not glob:
        glob = "*"
    if glob.startswith("/"):
        return glob
    return f"{prefix}{glob}"


def _normalize_match_pattern(pattern: str) -> str:
    normalized = pattern.strip()
    if normalized.startswith("/"):
        normalized = normalized[1:]
    return normalized


def _match_glob(pattern: str, path: str) -> bool:
    normalized_pattern = _normalize_match_pattern(pattern)
    normalized_path = path[1:] if path.startswith("/") else path
    path_obj = PurePosixPath(normalized_path)
    if path_obj.match(normalized_pattern):
        return True
    if normalized_pattern.startswith("**/") and "/" not in normalized_path:
        return path_obj.match(normalized_pattern[3:])
    return False


def _ensure_filesystem_columns(
    corpus,
    *,
    variant: str,
    path_builder: callable,
) -> None:
    attrs = getattr(corpus, "attrs", None)
    if attrs is None:
        raise ValueError("corpus must be a pandas DataFrame")
    indexed_variant = attrs.get("_filesystem_indexed")
    if indexed_variant:
        if indexed_variant != variant:
            raise ValueError(
                f"Filesystem already indexed for {indexed_variant}; cannot rebuild for {variant}."
            )
        return
    if "path" in corpus.columns:
        raise ValueError("Corpus already has a 'path' column; refusing to overwrite.")
    if "contents" in corpus.columns:
        raise ValueError("Corpus already has a 'contents' column; refusing to overwrite.")

    title_series = corpus.get("title")
    if title_series is None:
        title_series = pd.Series("", index=corpus.index)
    title_series = title_series.fillna("").astype(str)

    description_series = corpus.get("description")
    if description_series is None:
        description_series = pd.Series("", index=corpus.index)
    description_series = description_series.fillna("").astype(str)

    doc_id_series = corpus.get("doc_id")
    if doc_id_series is None:
        doc_id_series = pd.Series(corpus.index, index=corpus.index)
    doc_id_series = doc_id_series.fillna("").astype(str)

    path_series = path_builder(title_series, doc_id_series, corpus)
    if not isinstance(path_series, pd.Series):
        raise ValueError("path_builder must return a pandas Series")
    if path_series.isna().any():
        raise ValueError("path_builder produced empty paths")

    corpus["path"] = path_series
    corpus["contents"] = (
        "# "
        + title_series
        + " (ID: "
        + doc_id_series
        + ")\n\n"
        + description_series
    )
    attrs["_filesystem_indexed"] = variant


def _default_path_builder(
    title_series: pd.Series,
    doc_id_series: pd.Series,
    _corpus,
) -> pd.Series:
    title_slug = _slugify_series(title_series).mask(lambda s: s == "", "document")
    id_slug = _slugify_series(doc_id_series).mask(lambda s: s == "", doc_id_series)
    return "/" + title_slug + "-" + id_slug + ".txt"


def _wands_path_builder(
    title_series: pd.Series,
    doc_id_series: pd.Series,
    corpus,
) -> pd.Series:
    title_slug = _slugify_series(title_series).mask(lambda s: s == "", "document")
    id_slug = _slugify_series(doc_id_series).mask(lambda s: s == "", doc_id_series)
    base_name = title_slug + "-" + id_slug + ".txt"

    category_series = corpus.get("category")
    if category_series is None:
        category_series = pd.Series("", index=corpus.index)
    category_series = category_series.fillna("").astype(str)
    category_slug = _slugify_series(category_series)

    subcategory_series = corpus.get("subcategory")
    if subcategory_series is None:
        subcategory_series = pd.Series("", index=corpus.index)
    subcategory_series = subcategory_series.fillna("").astype(str)
    subcategory_slug = _slugify_series(subcategory_series)

    path = "/" + base_name
    has_category = category_slug != ""
    has_subcategory = subcategory_slug != ""
    path = path.where(~has_category, "/" + category_slug + "/" + base_name)
    path = path.where(
        ~(has_category & has_subcategory),
        "/" + category_slug + "/" + subcategory_slug + "/" + base_name,
    )
    return path


def _snippet_from_match(text: str, match: re.Match, window: int = 60) -> str:
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end]
    snippet = " ".join(snippet.splitlines()).strip()
    return snippet


def _make_filesystem_tools(corpus, *, variant: str, path_builder: callable):
    _ensure_filesystem_columns(corpus, variant=variant, path_builder=path_builder)
    path_series = corpus["path"].astype(str)
    contents_series = corpus["contents"].astype(str)
    paths = path_series.tolist()
    path_contents = list(zip(paths, contents_series.tolist()))
    timing_enabled = bool(os.getenv("EXPS_FS_TIMINGS"))
    path_index = None
    if not path_series.duplicated().any():
        path_index = dict(zip(paths, contents_series.tolist()))

    def ls(path: str, glob: str, max_results: int = 50) -> list[str]:
        """List files in a directory matching the glob, at most 50 results. Returns a list of paths."""
        if max_results > 50:
            raise ValueError("max_results must be <= 50")
        if max_results <= 0:
            return []
        prefix = _normalize_dir(path)
        pattern = _normalize_glob(prefix, glob)
        matches = []
        for item in paths:
            if not item.startswith(prefix):
                continue
            if _match_glob(pattern, item):
                matches.append(item)
                if len(matches) >= max_results:
                    break
        return sorted(matches)

    def grep(pattern: str, glob: str, num_results: int = 50) -> list[dict[str, str]]:
        """Search for a pattern in files matching the glob, at most 50 results."""
        if num_results > 50:
            raise ValueError("num_results must be <= 50")
        if num_results <= 0:
            return []
        compile_started = perf_counter() if timing_enabled else None
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {pattern}") from exc
        compile_ms = 0.0
        if timing_enabled:
            compile_ms = (perf_counter() - compile_started) * 1000
        match_pattern = _normalize_glob("/", glob)
        results = []
        scanned = 0
        glob_matched = 0
        regex_matched = 0
        scan_started = perf_counter() if timing_enabled else None
        for path, contents in path_contents:
            scanned += 1
            if not _match_glob(match_pattern, path):
                continue
            glob_matched += 1
            match = regex.search(contents)
            if not match:
                continue
            regex_matched += 1
            results.append({"path": path, "snippet": _snippet_from_match(contents, match)})
            if len(results) >= num_results:
                break
        if timing_enabled:
            scan_ms = (perf_counter() - scan_started) * 1000
            total_ms = compile_ms + scan_ms
            print(
                "[fs_grep_timing] pattern=%r glob=%r scanned=%d glob_matched=%d regex_matched=%d "
                "compile_ms=%.1f scan_ms=%.1f total_ms=%.1f"
                % (
                    pattern,
                    glob,
                    scanned,
                    glob_matched,
                    regex_matched,
                    compile_ms,
                    scan_ms,
                    total_ms,
                )
            )
        return results

    def cat(path: str) -> str:
        """Return the contents of a file as a string."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        if path_index is not None:
            if path not in path_index:
                raise ValueError(f"No file found for path: {path}")
            return str(path_index[path])
        matches = contents_series[path_series == path]
        if matches.empty:
            raise ValueError(f"No file found for path: {path}")
        if len(matches) > 1:
            raise ValueError(f"Multiple files found for path: {path}")
        return str(matches.iloc[0])

    return ls, grep, cat


def make_filesystem_ls_tool(corpus):
    return _make_filesystem_tools(corpus, variant="default", path_builder=_default_path_builder)[0]


def make_filesystem_grep_tool(corpus):
    return _make_filesystem_tools(corpus, variant="default", path_builder=_default_path_builder)[1]


def make_filesystem_cat_tool(corpus):
    return _make_filesystem_tools(corpus, variant="default", path_builder=_default_path_builder)[2]


def make_filesystem_ls_wands_tool(corpus):
    return _make_filesystem_tools(corpus, variant="wands", path_builder=_wands_path_builder)[0]


def make_filesystem_grep_wands_tool(corpus):
    return _make_filesystem_tools(corpus, variant="wands", path_builder=_wands_path_builder)[1]


def make_filesystem_cat_wands_tool(corpus):
    return _make_filesystem_tools(corpus, variant="wands", path_builder=_wands_path_builder)[2]
