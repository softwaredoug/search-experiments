from __future__ import annotations

from exps.tools import tool_kind
def resolve_id_column(df) -> str:
    for col in ("doc_id", "product_id", "id"):
        if col in df.columns:
            return col
    raise ValueError("Expected a doc identifier column (doc_id preferred).")


def build_id_lookup(corpus, id_col: str) -> dict[str, int]:
    return {str(doc_id): int(index) for index, doc_id in corpus[id_col].items()}


def load_rerank_fn(code: str, preferred_name: str | None) -> callable:
    exec_globals: dict = {}
    exec(code, exec_globals)
    if preferred_name and preferred_name in exec_globals:
        rerank_fn = exec_globals[preferred_name]
        if callable(rerank_fn):
            return rerank_fn
    if "reranker" in exec_globals and callable(exec_globals["reranker"]):
        return exec_globals["reranker"]
    for name, obj in exec_globals.items():
        if name.startswith("rerank_") and callable(obj):
            return obj
    raise ValueError("No rerank function found in code.")


def resolve_grade_column(judgments) -> str | None:
    for col in ("grade", "relevance", "rel", "label", "score"):
        if col in judgments.columns:
            return col
    return None


def split_search_tools(tool_config: list) -> tuple[list, list]:
    normal_tools: list = []
    raw_tools: list = []
    for entry in tool_config:
        if isinstance(entry, dict) and len(entry) == 1 and "raw" in entry:
            raw_entries = entry.get("raw") or []
            if not isinstance(raw_entries, list):
                raise ValueError("raw search tools must be a list")
            raw_tools.extend(raw_entries)
            continue
        if isinstance(entry, str):
            if tool_kind(entry) == "raw":
                raw_tools.append(entry)
            else:
                normal_tools.append(entry)
            continue
        if isinstance(entry, dict) and len(entry) == 1:
            tool_name = next(iter(entry))
            if tool_kind(tool_name) == "raw":
                raw_tools.append(entry)
            else:
                normal_tools.append(entry)
            continue
        normal_tools.append(entry)
    return normal_tools, raw_tools
