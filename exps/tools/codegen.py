from __future__ import annotations

import importlib.util
import inspect
import re
from pathlib import Path
from typing import Any, Union

import numpy as np


def _find_latest_reranker_path(path: Path) -> Path:
    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(f"codegen path not found: {path}")
    round_files = list(path.glob("reranker_round_*.py"))
    if round_files:
        def round_number(file_path: Path) -> int:
            match = re.search(r"reranker_round_(\d+)\.py", file_path.name)
            return int(match.group(1)) if match else -1

        return max(round_files, key=round_number)
    fallback = path / "reranker.py"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No reranker files found in {path}")


def _load_reranker_fn(path: Path, dataset_name: str | None) -> tuple[callable, str]:
    module_name = f"codegen_reranker_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Unable to load reranker module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if dataset_name:
        fn_name = f"rerank_{dataset_name}"
        reranker_fn = getattr(module, fn_name, None)
        if reranker_fn is not None:
            return reranker_fn, fn_name
    reranker_fn = getattr(module, "rerank", None)
    if reranker_fn is not None:
        return reranker_fn, "rerank"
    for name in sorted(module.__dict__):
        if not name.startswith("rerank_"):
            continue
        candidate = getattr(module, name)
        if callable(candidate):
            return candidate, name
    raise ValueError(f"No rerank function found in {path}")


def make_codegen_tool(
    corpus,
    *,
    tool_config: dict[str, Any],
    embeddings_device: str | None = None,
    dataset_name: str | None = None,
):
    from exps.tools.builder import build_search_tools

    path = tool_config.get("path")
    if not path:
        raise ValueError("codegen tool requires a path.")
    tool_name = tool_config.get("name") or "search"
    description = tool_config.get("description") or "Search the dataset and return results."
    return_fields = tool_config.get("return_fields") or []
    if not isinstance(return_fields, list):
        raise ValueError("return_fields must be a list when provided.")
    missing_fields = [field for field in return_fields if field not in corpus.columns]
    if missing_fields:
        missing_str = ", ".join(missing_fields)
        raise ValueError(f"return_fields not found in corpus: {missing_str}")
    dependencies = tool_config.get("dependencies") or []
    if not isinstance(dependencies, list):
        raise ValueError("dependencies must be a list of tool entries.")
    dependency_tools = build_search_tools(
        corpus,
        dependencies,
        embeddings_device=embeddings_device,
        dataset_name=dataset_name,
    )
    dependency_map = {tool.__name__: tool for tool in dependency_tools}
    reranker_path = _find_latest_reranker_path(Path(path).expanduser())
    reranker_fn, reranker_name = _load_reranker_fn(reranker_path, dataset_name)
    signature = inspect.signature(reranker_fn)
    params = list(signature.parameters.values())
    required_deps = []
    if params:
        for param in params[1:]:
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            required_deps.append(param.name)
    missing_deps = [dep for dep in required_deps if dep not in dependency_map]
    if missing_deps:
        missing_str = ", ".join(missing_deps)
        raise ValueError(f"codegen tool missing dependencies: {missing_str}")

    doc_lookup = corpus.set_index("doc_id", drop=False)

    def _to_builtin(value):
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _coerce_doc_id(value):
        value = _to_builtin(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return value
        return value

    def search(
        query: str,
        top_k: int = 10,
        agent_state=None,
        **kwargs,
    ) -> list[dict[str, Union[str, int, float]]]:
        """Search the corpus, return top results."""
        if top_k > 100:
            return "Error! top_k must be <= 100."
        reranker_results = reranker_fn(query=query, **dependency_map, **kwargs)
        results = []
        for rank, item in enumerate(reranker_results or []):
            if isinstance(item, dict):
                doc_id = _coerce_doc_id(item.get("id") or item.get("doc_id"))
                score = _to_builtin(item.get("score"))
            else:
                doc_id = _coerce_doc_id(item)
                score = None
            if doc_id is None or doc_id not in doc_lookup.index:
                continue
            row = doc_lookup.loc[doc_id]
            entry = {
                "id": int(_to_builtin(row.get("doc_id", doc_id))),
                "title": str(_to_builtin(row.get("title", ""))),
                "description": str(_to_builtin(row.get("description", ""))),
                "score": float(score) if score is not None else 1.0 / (rank + 1),
            }
            for field in return_fields:
                entry[field] = _to_builtin(row.get(field))
            results.append(entry)
            if len(results) >= top_k:
                break
        return results

    search.__name__ = tool_name
    search.__doc__ = description
    search.__codegen_reranker__ = reranker_name
    return search
