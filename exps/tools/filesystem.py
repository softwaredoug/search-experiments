from __future__ import annotations

import re
from fnmatch import fnmatch


def _slugify(value: str) -> str:
    lowered = value.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", lowered)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


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


def _format_contents(title: str, description: str) -> str:
    title_value = title or ""
    description_value = description or ""
    return f"Title: {title_value}\n\nDescription: {description_value}"


def _ensure_filesystem_columns(corpus) -> None:
    attrs = getattr(corpus, "attrs", None)
    if attrs is None:
        raise ValueError("corpus must be a pandas DataFrame")
    if attrs.get("_filesystem_indexed"):
        return
    if "path" in corpus.columns:
        raise ValueError("Corpus already has a 'path' column; refusing to overwrite.")
    if "contents" in corpus.columns:
        raise ValueError("Corpus already has a 'contents' column; refusing to overwrite.")

    paths = []
    contents = []
    for idx, row in corpus.iterrows():
        title = row.get("title", "")
        description = row.get("description", "")
        if title is None:
            title = ""
        if description is None:
            description = ""
        doc_id = row.get("doc_id", idx)
        if doc_id is None:
            doc_id = idx
        title_slug = _slugify(str(title)) or "document"
        id_slug = _slugify(str(doc_id)) or str(doc_id)
        path = f"/{title_slug}-{id_slug}.txt"
        paths.append(path)
        contents.append(_format_contents(str(title), str(description)))

    corpus["path"] = paths
    corpus["contents"] = contents
    attrs["_filesystem_indexed"] = True


def make_filesystem_ls_tool(corpus):
    _ensure_filesystem_columns(corpus)

    def ls(path: str, glob: str, max_results: int = 50) -> list[str]:
        """List files in a directory matching the glob, at most 50 results. Returns a list of paths."""
        if max_results > 50:
            raise ValueError("max_results must be <= 50")
        if max_results <= 0:
            return []
        prefix = _normalize_dir(path)
        pattern = _normalize_glob(prefix, glob)
        matches = []
        for item in corpus["path"].astype(str):
            if not item.startswith(prefix):
                continue
            if fnmatch(item, pattern):
                matches.append(item)
        return sorted(matches)[: max_results]

    return ls


def _snippet_from_match(text: str, match: re.Match, window: int = 60) -> str:
    start = max(0, match.start() - window)
    end = min(len(text), match.end() + window)
    snippet = text[start:end]
    snippet = " ".join(snippet.splitlines()).strip()
    return snippet


def make_filesystem_grep_tool(corpus):
    _ensure_filesystem_columns(corpus)

    def grep(pattern: str, glob: str, num_results: int = 50) -> list[dict[str, str]]:
        """Search for a pattern in files matching the glob, at most 50 results."""
        if num_results > 50:
            raise ValueError("num_results must be <= 50")
        if num_results <= 0:
            return []
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex pattern: {pattern}") from exc
        match_pattern = _normalize_glob("/", glob)
        results = []
        for path, contents in zip(corpus["path"].astype(str), corpus["contents"].astype(str)):
            if not fnmatch(path, match_pattern):
                continue
            match = regex.search(contents)
            if not match:
                continue
            results.append({"path": path, "snippet": _snippet_from_match(contents, match)})
            if len(results) >= num_results:
                break
        return results

    return grep


def make_filesystem_cat_tool(corpus):
    _ensure_filesystem_columns(corpus)

    def cat(path: str) -> str:
        """Return the contents of a file as a string."""
        if not isinstance(path, str) or not path.strip():
            raise ValueError("path must be a non-empty string")
        matches = corpus.loc[corpus["path"] == path, "contents"]
        if matches.empty:
            raise ValueError(f"No file found for path: {path}")
        if len(matches) > 1:
            raise ValueError(f"Multiple files found for path: {path}")
        return str(matches.iloc[0])

    return cat
