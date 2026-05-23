from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

from exps.codegen.utils import load_rerank_fn
from exps.tools import build_search_tools, normalize_search_tools


@dataclass
class SearchToolConfig:
    name: str
    doc: str
    fn: callable
    kind: str
    is_primary: bool = False


@dataclass
class SearchToolState:
    tools: list[SearchToolConfig]
    search_tools: list[SearchToolConfig]
    raw_tools: list[SearchToolConfig]
    primary_tool: SearchToolConfig
    tool_params: list[str]


def _validate_start_code(code: str, rerank_name: str, tool_fns: list[callable]) -> None:
    try:
        rerank_fn = load_rerank_fn(code, rerank_name)
    except Exception as exc:
        raise ValueError(f"start_code must define a callable {rerank_name} function: {exc}") from exc
    try:
        signature = inspect.signature(rerank_fn)
        if "top_k" in signature.parameters:
            rerank_fn("test query", top_k=10, *tool_fns)
        else:
            rerank_fn("test query", *tool_fns)
    except Exception as exc:
        raise ValueError(
            "start_code does not match configured tools; verify the rerank signature and search_tools."
        ) from exc


def build_search_tool_state(
    *,
    corpus,
    dataset: str,
    device: str | None,
    normal_tool_config: list,
    raw_tool_config: list,
    rerank_name: str,
    code_path: Path,
    start_code_from_config: bool,
) -> SearchToolState:
    search_tool_fns = build_search_tools(
        corpus,
        normal_tool_config,
        embeddings_device=device,
        dataset_name=dataset,
    )
    raw_tool_fns = build_search_tools(
        corpus,
        raw_tool_config,
        embeddings_device=device,
        dataset_name=dataset,
        context="raw",
    )
    if not search_tool_fns and not raw_tool_fns:
        raise ValueError("Codegen requires at least one search tool.")
    tool_fns = search_tool_fns + raw_tool_fns
    if start_code_from_config:
        _validate_start_code(code_path.read_text(encoding="utf-8"), rerank_name, tool_fns)

    search_tools = [
        SearchToolConfig(
            name=tool.__name__,
            doc=tool.__doc__ or "",
            fn=tool,
            kind="search",
        )
        for tool in search_tool_fns
    ]
    raw_tools = [
        SearchToolConfig(
            name=tool.__name__,
            doc=tool.__doc__ or "",
            fn=tool,
            kind="raw",
        )
        for tool in raw_tool_fns
    ]
    tools = search_tools + raw_tools
    if search_tools:
        tools[0].is_primary = True
        primary_tool = tools[0]
    else:
        raw_tools[0].is_primary = True
        primary_tool = raw_tools[0]
    tool_params = [tool.name for tool in tools]
    return SearchToolState(
        tools=tools,
        search_tools=search_tools,
        raw_tools=raw_tools,
        primary_tool=primary_tool,
        tool_params=tool_params,
    )


def normalize_tool_config(tool_config: list) -> list[dict]:
    return normalize_search_tools(tool_config)
