from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Union

import nbformat
import yaml
from typing_extensions import Literal

from exps.tools.registry import TOOL_REGISTRY

NOTEBOOKS_DIR = Path("notebooks")

NOTEBOOK_CONFIG_PAIRS = [
    ("notebooks/bm25_ms_marco.ipynb", "configs/ecom_base/bm25.yml"),
    (
        "notebooks/agentic_ecom_2tools_wands.ipynb",
        "configs/ecom_base/agentic_ecom_2tools_gpt5_mini.yml",
    ),
    ("notebooks/agentic_msmarco_minimarco.ipynb", "configs/agentic_msmarco.yml"),
]


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _resolve_tool_name(function_name: str) -> str | None:
    if function_name in TOOL_REGISTRY:
        return function_name
    if function_name.startswith("search_"):
        candidate = function_name[len("search_") :]
        if candidate in TOOL_REGISTRY:
            return candidate
    return None


def _load_notebook_functions(path: Path) -> dict[str, callable]:
    notebook = nbformat.read(path, as_version=4)
    found: dict[str, callable] = {}
    for cell in notebook.cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or "")
        if not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if _resolve_tool_name(node.name) is None:
                continue
            func_source = ast.get_source_segment(source, node)
            if not func_source:
                continue
            namespace = {"Literal": Literal, "Union": Union}
            exec("from __future__ import annotations\n" + func_source, namespace)
            found[node.name] = namespace[node.name]
    return found


def _extract_returned_inner_function(builder: callable) -> ast.FunctionDef | None:
    source = inspect.getsource(builder)
    module = ast.parse(source)
    builder_node = next((node for node in module.body if isinstance(node, ast.FunctionDef)), None)
    if builder_node is None:
        return None
    returned_name: str | None = None
    for stmt in builder_node.body:
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Name):
            returned_name = stmt.value.id
    if returned_name is None:
        return None
    for stmt in builder_node.body:
        if isinstance(stmt, ast.FunctionDef) and stmt.name == returned_name:
            return stmt
    return None


def _tool_from_registry(tool_name: str) -> callable | None:
    entry = TOOL_REGISTRY.get(tool_name)
    if entry is None:
        return None
    builder = entry["builder"]
    inner_fn_node = _extract_returned_inner_function(builder)
    if inner_fn_node is None:
        return None
    source = inspect.getsource(builder)
    func_source = ast.get_source_segment(source, inner_fn_node)
    if not func_source:
        return None
    future_header = "from __future__ import annotations\n"
    namespace: dict[str, Any] = {"Literal": Literal, "Union": Union}
    exec(future_header + func_source, namespace)
    return namespace[inner_fn_node.name]


def _format_signature(sig: inspect.Signature) -> str:
    formatted_params = []
    for param in sig.parameters.values():
        annotation = inspect.formatannotation(param.annotation)
        default = "" if param.default is inspect._empty else f"={repr(param.default)}"
        prefix = ""
        if param.kind == param.VAR_POSITIONAL:
            prefix = "*"
        elif param.kind == param.VAR_KEYWORD:
            prefix = "**"
        name = f"{prefix}{param.name}"
        if annotation and annotation != "_empty":
            name = f"{name}: {annotation}"
        formatted_params.append(f"{name}{default}")
    return_annotation = inspect.formatannotation(sig.return_annotation)
    suffix = ""
    if return_annotation and return_annotation != "_empty":
        suffix = f" -> {return_annotation}"
    return f"({', '.join(formatted_params)}){suffix}"


def _raw_tool_names() -> set[str]:
    return {name for name, entry in TOOL_REGISTRY.items() if entry.get("kind") == "raw"}


def _extract_list_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    if not isinstance(node, ast.List):
        return names
    for item in node.elts:
        if isinstance(item, ast.Name):
            names.append(item.id)
    return names


def _find_agent_tool_lists(path: Path) -> list[tuple[int, list[str]]]:
    notebook = nbformat.read(path, as_version=4)
    tool_lists: list[tuple[int, list[str]]] = []
    for cell_index, cell in enumerate(notebook.cells):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source") or "")
        if not source.strip():
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == "tools" for target in node.targets):
                    names = _extract_list_names(node.value)
                    if names:
                        tool_lists.append((cell_index, names))
            if isinstance(node, ast.Call):
                func_name = None
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                if func_name != "OpenAIAgent":
                    continue
                for keyword in node.keywords:
                    if keyword.arg == "tools":
                        names = _extract_list_names(keyword.value)
                        if names:
                            tool_lists.append((cell_index, names))
    return tool_lists


def test_notebook_first_cell_includes_config_description():
    for notebook_path, config_path in NOTEBOOK_CONFIG_PAIRS:
        config_file = Path(config_path)
        if not config_file.exists():
            continue
        config = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        description = config.get("strategy", {}).get("description")
        if not description:
            continue

        nb = nbformat.read(Path(notebook_path), as_version=4)
        assert nb.cells, f"Notebook has no cells: {notebook_path}"
        first = nb.cells[0]
        assert first.get("cell_type") == "markdown", (
            f"First cell must be markdown in {notebook_path}"
        )
        first_text = "".join(first.get("source", []))
        assert _normalize(description) in _normalize(first_text), (
            f"Notebook {notebook_path} first cell missing description from {config_path}"
        )


def test_notebook_tools_match_repo_inventory():
    notebook_paths = [
        path
        for path in NOTEBOOKS_DIR.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    ]
    assert notebook_paths, "No notebooks found in notebooks/"

    for notebook_path in notebook_paths:
        notebook_tools = _load_notebook_functions(notebook_path)
        for name, notebook_tool in notebook_tools.items():
            tool_name = _resolve_tool_name(name)
            if tool_name is None:
                continue
            repo_tool = _tool_from_registry(tool_name)
            assert repo_tool is not None, f"Tool {tool_name} missing from tool registry."
            assert notebook_tool.__name__ == repo_tool.__name__, f"{notebook_path}:{name} name mismatch"
            notebook_sig = _format_signature(inspect.signature(notebook_tool))
            repo_sig = _format_signature(inspect.signature(repo_tool))
            assert notebook_sig == repo_sig, f"{notebook_path}:{name} signature mismatch"
            assert inspect.getdoc(notebook_tool) == inspect.getdoc(
                repo_tool
            ), f"{notebook_path}:{name} docstring mismatch"


def test_notebook_agents_do_not_receive_raw_tools():
    raw_tools = _raw_tool_names()
    assert raw_tools, "No raw tools registered in TOOL_REGISTRY"
    notebook_paths = [
        path
        for path in NOTEBOOKS_DIR.rglob("*.ipynb")
        if ".ipynb_checkpoints" not in path.parts
    ]
    assert notebook_paths, "No notebooks found in notebooks/"

    violations: list[str] = []
    for notebook_path in notebook_paths:
        for cell_index, tool_names in _find_agent_tool_lists(notebook_path):
            bad = sorted(set(tool_names) & raw_tools)
            if bad:
                violations.append(
                    f"{notebook_path}:cell_{cell_index + 1} includes raw tools: {', '.join(bad)}"
                )
    assert not violations, "\n".join(violations)
