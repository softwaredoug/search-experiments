from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import PurePosixPath
from time import perf_counter

import pandas as pd
from cheat_at_search.agent.openai_agent import OpenAIAgent


FILESYSTEM_ROOT_KEY = "filesystem_root"
SEARCH_DIRECTORY_DEPTH_KEY = "search_directory_depth"


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


def _build_filename(title_slug: str, id_slug: str, *, max_len: int = 200) -> str:
    base = f"{title_slug}-{id_slug}.txt"
    if len(base) <= max_len:
        return base
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:8]
    suffix = f"-{digest}-{id_slug}.txt"
    max_title_len = max_len - len(suffix)
    if max_title_len < 1:
        truncated_title = title_slug[:1] if title_slug else "d"
        return f"{truncated_title}{suffix}"
    truncated_title = title_slug[:max_title_len]
    return f"{truncated_title}{suffix}"


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


def _normalize_path(path: str) -> str:
    trimmed = path.strip().strip('"').strip("'")
    if not trimmed:
        return ""
    if trimmed.startswith("./"):
        trimmed = trimmed[2:]
    if not trimmed.startswith("/"):
        trimmed = f"/{trimmed}"
    while "//" in trimmed:
        trimmed = trimmed.replace("//", "/")
    return trimmed


def _normalize_root(path: str | None) -> str:
    if not path:
        return "/"
    normalized = _normalize_path(path)
    if not normalized:
        return "/"
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return normalized


def _contains_path_escape(path: str) -> bool:
    return ".." in PurePosixPath(path).parts


def _is_under_root(path: str, root: str) -> bool:
    if root == "/":
        return True
    return path == root or path.startswith(f"{root}/")


def _join_under_root(root: str, path: str) -> str:
    if path in {"", ".", "./"}:
        return root
    if path.startswith("/"):
        return _normalize_path(path)
    if root == "/":
        return _normalize_path(path)
    return _normalize_path(f"{root}/{path}")


def _filesystem_root(agent_state: dict | None) -> str:
    if agent_state is None:
        return "/"
    return _normalize_root(agent_state.get(FILESYSTEM_ROOT_KEY))


def _resolve_scoped_path(path: str, agent_state: dict | None) -> tuple[str | None, str | None]:
    if not isinstance(path, str) or not path.strip():
        return None, "Error! path must be a non-empty string."
    root = _filesystem_root(agent_state)
    if _contains_path_escape(path):
        return None, f"Error! path is outside filesystem root: {root}"
    resolved = _join_under_root(root, path.strip())
    if not _is_under_root(resolved, root):
        return None, f"Error! path is outside filesystem root: {root}"
    return resolved, None


def _resolve_scoped_glob(glob: str, agent_state: dict | None) -> tuple[str | None, str | None]:
    if not isinstance(glob, str):
        return None, "Error! glob must be a string."
    root = _filesystem_root(agent_state)
    glob = glob or "*"
    if _contains_path_escape(glob):
        return None, f"Error! glob is outside filesystem root: {root}"
    if glob.startswith("/"):
        resolved = _normalize_path(glob)
        if not _is_under_root(resolved, root):
            return None, f"Error! glob is outside filesystem root: {root}"
        return resolved, None
    if root == "/":
        return glob, None
    return f"{root}/{glob}", None


def _match_glob(pattern: str, path: str) -> bool:
    normalized_pattern = _normalize_match_pattern(pattern)
    normalized_path = path[1:] if path.startswith("/") else path
    path_obj = PurePosixPath(normalized_path)
    if path_obj.match(normalized_pattern):
        return True
    if normalized_pattern.startswith("**/") and "/" not in normalized_path:
        return path_obj.match(normalized_pattern[3:])
    if "/**/" in normalized_pattern:
        shallow_pattern = normalized_pattern.replace("/**/", "/")
        if path_obj.match(shallow_pattern):
            return True
    return False


def _append_tool_output(results: list[dict], output) -> None:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                results.append(item)
        return
    if isinstance(output, dict):
        results.append(output)


def _collect_tool_outputs(items: list[dict]) -> list[dict]:
    results: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call_output":
            continue
        _append_tool_output(results, item.get("output"))
    return results


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
    filenames = [
        _build_filename(title, doc_id)
        for title, doc_id in zip(title_slug.tolist(), id_slug.tolist())
    ]
    return pd.Series([f"/{name}" for name in filenames], index=title_series.index)


def _wands_path_builder(
    title_series: pd.Series,
    doc_id_series: pd.Series,
    corpus,
) -> pd.Series:
    title_slug = _slugify_series(title_series).mask(lambda s: s == "", "document")
    id_slug = _slugify_series(doc_id_series).mask(lambda s: s == "", doc_id_series)
    base_name = pd.Series(
        [
            _build_filename(title, doc_id)
            for title, doc_id in zip(title_slug.tolist(), id_slug.tolist())
        ],
        index=title_series.index,
    )

    category_series = corpus.get("category")
    if category_series is None:
        category_series = pd.Series("", index=corpus.index)
    category_series = category_series.fillna("").astype(str)
    category_slug = _slugify_series(category_series)

    subcategory_series = corpus.get("subcategory")
    if subcategory_series is None and "sub_category" in corpus.columns:
        print("Debug: using sub_category column for WANDS filesystem paths")
        subcategory_series = corpus.get("sub_category")
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


def _make_filesystem_tools(
    corpus,
    *,
    variant: str,
    path_builder: callable,
    model: str = "gpt-5-mini",
    reasoning: str = "low",
    system_prompt: str | None = None,
):
    _ensure_filesystem_columns(corpus, variant=variant, path_builder=path_builder)
    path_series = corpus["path"].astype(str)
    contents_series = corpus["contents"].astype(str)
    paths = path_series.tolist()
    path_contents = list(zip(paths, contents_series.tolist()))
    timing_enabled = bool(os.getenv("EXPS_FS_TIMINGS"))
    path_index = None
    if not path_series.duplicated().any():
        path_index = dict(zip(paths, contents_series.tolist()))

    wands_doc = None
    if variant == "wands":
        wands_doc = (
            "WANDS filesystem layout uses <category>/<subcategory>/<product-name-slug>-<doc-id>.txt. "
            "Example: /Furniture/Armchairs/sancroft-armchair-1234.txt. "
            "File contents are:\n\n"
            "<Title> (ID: <ID>)\n\n<Description>\n\n"
            "Example file contents:\n\n"
            "Sancroft Armchair (ID: 1234)\n\n"
            "A compact armchair with tailored upholstery, a supportive back, "
            "and gently flared arms designed for small spaces."
        )

    def ls(path: str, glob: str, max_results: int = 50, agent_state=None) -> list[str] | str:
        """List files in a directory matching the glob, at most 50 results. Returns a list of paths."""
        limit = max_results
        if max_results > 50:
            limit = 50
        if max_results <= 0:
            return []
        scoped_path, error = _resolve_scoped_path(path, agent_state)
        if error:
            return error
        scoped_glob, error = _resolve_scoped_glob(glob, agent_state)
        if error:
            return error
        prefix = _normalize_dir(scoped_path)
        if glob in {"*", "*/"}:
            child_map: dict[str, bool] = {}
            for item in paths:
                if not item.startswith(prefix):
                    continue
                rest = item[len(prefix):]
                if not rest:
                    continue
                child = rest.split("/", 1)[0]
                child_path = f"{prefix}{child}" if prefix != "/" else f"/{child}"
                is_dir = "/" in rest
                if child_path in child_map:
                    child_map[child_path] = child_map[child_path] or is_dir
                else:
                    child_map[child_path] = is_dir
            dir_children = sorted([path for path, is_dir in child_map.items() if is_dir])
            file_children = sorted([path for path, is_dir in child_map.items() if not is_dir])
            ordered = dir_children + file_children
            if len(ordered) > limit:
                extra = len(ordered) - limit
                ordered = ordered[:limit]
                ordered.append(f"Truncated ({extra} more)")
            return ordered
        pattern = _normalize_glob(prefix, scoped_glob)
        matches = []
        extra = 0
        for item in paths:
            if not item.startswith(prefix):
                continue
            if _match_glob(pattern, item):
                if len(matches) < limit:
                    matches.append(item)
                else:
                    extra += 1
        matches = sorted(matches)
        if extra:
            matches.append(f"Truncated ({extra} more)")
        return matches

    def grep(
        pattern: str,
        glob: str,
        num_results: int = 50,
        agent_state=None,
    ) -> list[dict[str, str]] | str:
        """Search for a regex pattern in files matching the glob, at most 50 results."""
        limit = num_results
        if num_results > 50:
            limit = 50
        if num_results <= 0:
            return []
        compile_started = perf_counter() if timing_enabled else None
        try:
            regex = re.compile(pattern)
        except re.error:
            return f"Error! Invalid regex pattern: {pattern}"
        compile_ms = 0.0
        if timing_enabled:
            compile_ms = (perf_counter() - compile_started) * 1000
        scoped_glob, error = _resolve_scoped_glob(glob, agent_state)
        if error:
            return error
        match_pattern = _normalize_glob("/", scoped_glob)
        results = []
        scanned = 0
        glob_matched = 0
        regex_matched = 0
        extra = 0
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
            if len(results) < limit:
                results.append({"path": path, "snippet": _snippet_from_match(contents, match)})
            else:
                extra += 1
        if timing_enabled and agent_state is not None:
            scan_ms = (perf_counter() - scan_started) * 1000
            total_ms = compile_ms + scan_ms
            logger = agent_state.get("trace_logger") if agent_state else None
            if logger is not None:
                logger.info(
                    "fs_grep_timing %s",
                    {
                        "pattern": pattern,
                        "glob": glob,
                        "scanned": scanned,
                        "glob_matched": glob_matched,
                        "regex_matched": regex_matched,
                        "compile_ms": round(compile_ms, 1),
                        "scan_ms": round(scan_ms, 1),
                        "total_ms": round(total_ms, 1),
                    },
                )
        if extra:
            results.append({"path": "", "snippet": f"Truncated ({extra} more)"})
        return results

    def cat(path: str, agent_state=None) -> str:
        """Return the contents of a file as a string."""
        normalized_path, error = _resolve_scoped_path(path, agent_state)
        if error:
            return error
        if path_index is not None:
            if normalized_path in path_index:
                return str(path_index[normalized_path])
            if path in path_index:
                return str(path_index[path])
        matches = contents_series[path_series == normalized_path]
        if matches.empty and normalized_path != path:
            matches = contents_series[path_series == path]
        if matches.empty:
            filename = PurePosixPath(normalized_path or path).name
            if filename:
                filename_matches = contents_series[path_series.str.endswith(f"/{filename}")]
                if len(filename_matches) == 1:
                    return str(filename_matches.iloc[0])
                if len(filename_matches) > 1:
                    return f"Error! Multiple files found for filename: {filename}"
            return f"Error! No file found for path: {path}"
        if len(matches) > 1:
            return f"Error! Multiple files found for path: {path}"
        return str(matches.iloc[0])

    def search_directory(directory: str, prompt: str, agent_state=None) -> list[dict] | str:
        """Delegate search within a specific directory to a sub-agent."""
        if agent_state is None:
            agent_state = {}
        depth = int(agent_state.get(SEARCH_DIRECTORY_DEPTH_KEY, 0)) + 1
        if depth > 1:
            return "Error! search_directory cannot be nested."
        scoped_directory, error = _resolve_scoped_path(directory, agent_state)
        if error:
            return error
        scoped_directory = _normalize_root(scoped_directory)
        subagent_state = dict(agent_state)
        subagent_state[FILESYSTEM_ROOT_KEY] = scoped_directory
        subagent_state[SEARCH_DIRECTORY_DEPTH_KEY] = depth
        logger = agent_state.get("trace_logger")
        if logger is not None:
            logger.info("search_directory_call %s", {"directory": scoped_directory, "prompt": prompt})
        subagent_prompt = system_prompt or (
            "You search a virtual filesystem rooted at {scope}. Use ls, grep, and cat to find relevant files."
        )
        subagent_prompt = subagent_prompt.replace("{scope}", scoped_directory)
        agent = OpenAIAgent(
            tools=[ls, grep, cat],
            model=f"openai/{model}" if "/" not in model else model,
            reasoning_level=reasoning,
            response_model=None,
        )
        inputs = [
            {"role": "system", "content": subagent_prompt},
            {
                "role": "user",
                "content": (
                    f"Directory scope: {scoped_directory}\n"
                    f"Task: {prompt}\n"
                    "Use only the available filesystem tools. Return useful paths and snippets."
                ),
            },
        ]
        previous_inputs = list(inputs)
        _, inputs, _ = agent.chat(inputs=inputs, agent_state=subagent_state, logger=logger)
        results = _collect_tool_outputs(inputs[len(previous_inputs):])
        if logger is not None:
            logger.info("search_directory_result %s", {"directory": scoped_directory, "result_count": len(results)})
        return results

    if wands_doc:
        ls.__doc__ = f"{ls.__doc__}\n\n{wands_doc}"
        grep.__doc__ = f"{grep.__doc__}\n\n{wands_doc}"
        cat.__doc__ = f"{cat.__doc__}\n\n{wands_doc}"
    if variant == "wands":
        ls.__name__ = "ls_wands"
        grep.__name__ = "grep_wands"
        cat.__name__ = "cat_wands"
        search_directory.__name__ = "search_directory_wands"

    return ls, grep, cat, search_directory


def make_filesystem_ls_tool(corpus):
    return _make_filesystem_tools(corpus, variant="default", path_builder=_default_path_builder)[0]


def make_filesystem_grep_tool(corpus):
    return _make_filesystem_tools(corpus, variant="default", path_builder=_default_path_builder)[1]


def make_filesystem_cat_tool(corpus):
    return _make_filesystem_tools(corpus, variant="default", path_builder=_default_path_builder)[2]


def make_filesystem_search_directory_tool(
    corpus,
    *,
    model: str = "gpt-5-mini",
    reasoning: str = "low",
    system_prompt: str | None = None,
    **_unused,
):
    return _make_filesystem_tools(
        corpus,
        variant="default",
        path_builder=_default_path_builder,
        model=model,
        reasoning=reasoning,
        system_prompt=system_prompt,
    )[3]


def make_filesystem_ls_wands_tool(corpus):
    return _make_filesystem_tools(corpus, variant="wands", path_builder=_wands_path_builder)[0]


def make_filesystem_grep_wands_tool(corpus):
    return _make_filesystem_tools(corpus, variant="wands", path_builder=_wands_path_builder)[1]


def make_filesystem_cat_wands_tool(corpus):
    return _make_filesystem_tools(corpus, variant="wands", path_builder=_wands_path_builder)[2]


def make_filesystem_search_directory_wands_tool(
    corpus,
    *,
    model: str = "gpt-5-mini",
    reasoning: str = "low",
    system_prompt: str | None = None,
    **_unused,
):
    return _make_filesystem_tools(
        corpus,
        variant="wands",
        path_builder=_wands_path_builder,
        model=model,
        reasoning=reasoning,
        system_prompt=system_prompt,
    )[3]
