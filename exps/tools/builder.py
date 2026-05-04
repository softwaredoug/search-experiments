from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from exps.tools.guards import GUARDS
from exps.tools.registry import TOOL_REGISTRY, tool_kind


def _parse_guard_entry(entry: Any) -> tuple[str, dict]:
    if isinstance(entry, str):
        return entry, {}
    if isinstance(entry, dict) and len(entry) == 1:
        name = next(iter(entry))
        params = entry[name]
        if params is None:
            params = {}
        if not isinstance(params, dict):
            if name == "disallow_similar_queries":
                return name, {"threshold": params}
            raise ValueError(f"Guard params for {name} must be a mapping.")
        return name, params
    raise ValueError("Guard entry must be a string or single-key mapping.")


def _guard_doc(guard: dict) -> str:
    guard_name = guard["name"]
    guard_params = guard.get("params", {})
    guard_fn = GUARDS.get(guard_name)
    description = ""
    if guard_fn and guard_fn.__doc__:
        description = guard_fn.__doc__.strip()
    if guard_params:
        params_str = ", ".join(f"{key}={value}" for key, value in guard_params.items())
        if description:
            return f"{guard_name}({params_str}): {description}"
        return f"{guard_name}({params_str})"
    if description:
        return f"{guard_name}: {description}"
    return guard_name


def _normalize_tool_config(tool_info: dict) -> dict[str, Any]:
    return {key: value for key, value in tool_info.items() if key != "guards"}


def normalize_search_tools(tool_config: list) -> list[dict[str, Any]]:
    normalized = []
    for item in tool_config:
        if isinstance(item, dict) and len(item) == 1 and "raw" in item:
            raise ValueError("raw search tools are only supported in codegen configs.")
        if isinstance(item, str):
            normalized.append({"name": item, "guards": [], "config": {}})
            continue
        if isinstance(item, dict) and len(item) == 1:
            tool_name = next(iter(item))
            tool_info = item[tool_name] or {}
            if not isinstance(tool_info, dict):
                raise ValueError(f"Tool config for {tool_name} must be a mapping.")
            guards_raw = tool_info.get("guards") or []
            if not isinstance(guards_raw, list):
                raise ValueError(f"Guards for {tool_name} must be a list.")
            guards = []
            for guard_entry in guards_raw:
                guard_name, guard_params = _parse_guard_entry(guard_entry)
                guards.append({"name": guard_name, "params": guard_params})
            normalized.append(
                {
                    "name": tool_name,
                    "guards": guards,
                    "config": _normalize_tool_config(tool_info),
                }
            )
            continue
        raise ValueError("search_tools entries must be strings or single-key mappings.")
    return normalized


def _normalize_tool_config_for_cache(config: dict[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in config.items():
        if key == "dependencies":
            if not isinstance(value, list):
                raise ValueError("dependencies must be a list of tool entries.")
            normalized[key] = normalize_search_tools_for_cache(value)
            continue
        if isinstance(value, Path):
            normalized[key] = str(value)
            continue
        normalized[key] = value
    return dict(sorted(normalized.items()))


def normalize_search_tools_for_cache(tool_config: list) -> list[dict[str, Any]]:
    normalized = normalize_search_tools(tool_config)
    normalized_sorted = []
    for tool in sorted(normalized, key=lambda item: item["name"]):
        guards = sorted(tool["guards"], key=lambda item: item["name"])
        guards = [
            {"name": guard["name"], "params": dict(sorted(guard["params"].items()))}
            for guard in guards
        ]
        config = tool.get("config") or {}
        normalized_sorted.append(
            {
                "name": tool["name"],
                "guards": guards,
                "config": _normalize_tool_config_for_cache(config),
            }
        )
    return normalized_sorted


def make_guarded_search_tool(
    tool_fn: callable,
    *,
    guards: list[dict[str, Any]] | None = None,
    func_name: Optional[str] = None,
):
    name = func_name or tool_fn.__name__
    guards = guards or []

    def guarded(
        query: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict[str, Union[str, int, float]]] | str:
        """Search tool wrapper that enforces configured guard checks."""
        if top_k > 20:
            return "Error! top_k must be <= 20."
        params = {
            "tool_name": name,
            "query": query,
            "top_k": top_k,
        }
        for guard in guards:
            guard_name = guard["name"]
            guard_params = guard.get("params", {})
            guard_fn = GUARDS.get(guard_name)
            if guard_fn is None:
                raise ValueError(f"Unknown guard: {guard_name}")
            err = guard_fn(params, agent_state, **guard_params)
            if isinstance(err, str) and err:
                return err
        return tool_fn(query, top_k, agent_state)

    guarded.__name__ = name
    guarded.__doc__ = tool_fn.__doc__
    return guarded


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


def build_search_tools(
    corpus,
    tool_config: list,
    embeddings_device: str | None = None,
    dataset_name: str | None = None,
    context: str = "agentic",
):
    tools = []
    for tool in normalize_search_tools(tool_config):
        tool_name = tool["name"]
        entry = TOOL_REGISTRY.get(tool_name)
        if entry is None:
            raise ValueError(f"Unknown search tool: {tool_name}")
        if context == "agentic" and entry.get("kind", "agentic") == "raw":
            raise ValueError(f"{tool_name} is a raw search tool and cannot be used in agentic strategies.")
        if tool_name.endswith("_wands"):
            if dataset_name != "wands":
                raise ValueError(f"{tool_name} is only available for wands dataset.")
        builder = entry["builder"]
        if tool_name in {"codegen", "query_rewrite"}:
            tool_fn = builder(
                corpus,
                tool_config=tool.get("config") or {},
                embeddings_device=embeddings_device,
                dataset_name=dataset_name,
            )
        elif tool_name == "bm25":
            tool_params = tool.get("config", {}).get("params", {})
            tool_fn = builder(corpus, **tool_params)
        elif tool_name in {"embeddings", "minilm", "e5_base_v2"}:
            tool_fn = builder(corpus, device=embeddings_device)
        else:
            tool_fn = builder(corpus)
        if tool["guards"]:
            guard_lines = ["Guards:"]
            guard_lines.extend(f"- {_guard_doc(guard)}" for guard in tool["guards"])
            guard_block = "\n".join(guard_lines)
            base_doc = tool_fn.__doc__ or ""
            tool_fn.__doc__ = f"{base_doc}\n\n{guard_block}" if base_doc else guard_block
        if tool["guards"]:
            if tool_name == "query_rewrite":
                raise ValueError("query_rewrite does not support guards.")
            if tool_name == "codegen":
                guard_func_name = tool_fn.__name__
            else:
                guard_func_name = f"search_{tool_name}"
            tool_fn = make_guarded_search_tool(
                tool_fn,
                guards=tool["guards"],
                func_name=guard_func_name,
            )
        tools.append(tool_fn)
    return tools
