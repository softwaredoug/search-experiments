from __future__ import annotations

from typing import Any

import pandas as pd


def _present(value: Any) -> bool:
    if value is None:
        return False
    missing = pd.isna(value)
    return not bool(missing)


def format_corpus_result(row, *, score: Any = 0.0, fallback_id: Any = None) -> dict:
    result = {
        "id": row.get("doc_id", fallback_id),
        "title": row.get("title", ""),
        "description": row.get("description", ""),
        "score": score,
    }
    if "path" in row.index:
        result["path"] = row.get("path", "")
    if "image_url" in row.index and _present(row.get("image_url")):
        result["image_url"] = row.get("image_url")
    return result
