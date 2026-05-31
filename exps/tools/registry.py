from __future__ import annotations

from exps.tools.bm25 import make_bm25_tool, make_fielded_bm25_tool
from exps.tools.bash_tool import make_bash_tool, make_bash_wands_tool
from exps.tools.codegen import make_codegen_tool
from exps.tools.embeddings import make_embedding_tool
from exps.tools.filesystem import (
    make_filesystem_cat_tool,
    make_filesystem_cat_wands_tool,
    make_filesystem_grep_tool,
    make_filesystem_grep_wands_tool,
    make_filesystem_ls_tool,
    make_filesystem_ls_wands_tool,
    make_filesystem_search_directory_tool,
    make_filesystem_search_directory_wands_tool,
)
from exps.tools.query_rewrite import make_query_rewrite_tool
from exps.tools.raw import make_get_corpus_tool
from exps.tools.todo import make_todo_read_tool, make_todo_write_tool
from exps.tools.wands import (
    make_check_features_wands_tool,
    make_top_categories_tool,
    make_wands_bm25_tool,
    make_wands_bm25_prefiltered_tool,
    make_wands_embedding_tool,
    make_wands_embedding_prefiltered_tool,
)

TOOL_REGISTRY = {
    "bm25": {"builder": make_bm25_tool, "kind": "agentic"},
    "fielded_bm25": {"builder": make_fielded_bm25_tool, "kind": "agentic"},
    "bash": {"builder": make_bash_tool, "kind": "agentic"},
    "bash_wands": {"builder": make_bash_wands_tool, "kind": "agentic"},
    "minilm": {"builder": make_embedding_tool, "kind": "agentic"},
    "embeddings": {"builder": make_embedding_tool, "kind": "agentic"},
    "codegen": {"builder": make_codegen_tool, "kind": "agentic"},
    "query_rewrite": {"builder": make_query_rewrite_tool, "kind": "agentic"},
    "todo_write": {"builder": make_todo_write_tool, "kind": "agentic"},
    "todo_read": {"builder": make_todo_read_tool, "kind": "agentic"},
    "get_corpus": {"builder": make_get_corpus_tool, "kind": "raw"},
    "ls": {"builder": make_filesystem_ls_tool, "kind": "agentic"},
    "grep": {"builder": make_filesystem_grep_tool, "kind": "agentic"},
    "cat": {"builder": make_filesystem_cat_tool, "kind": "agentic"},
    "search_directory": {"builder": make_filesystem_search_directory_tool, "kind": "agentic"},
    "ls_wands": {"builder": make_filesystem_ls_wands_tool, "kind": "agentic"},
    "grep_wands": {"builder": make_filesystem_grep_wands_tool, "kind": "agentic"},
    "cat_wands": {"builder": make_filesystem_cat_wands_tool, "kind": "agentic"},
    "search_directory_wands": {"builder": make_filesystem_search_directory_wands_tool, "kind": "agentic"},
    "e5_base_v2": {
        "builder": lambda corpus, device=None: make_embedding_tool(
            corpus,
            device=device,
            model_name="intfloat/e5-base-v2",
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
        "kind": "agentic",
    },
    "bm25_wands": {"builder": make_wands_bm25_tool, "kind": "agentic"},
    "bm25_wands_prefiltered": {
        "builder": make_wands_bm25_prefiltered_tool,
        "kind": "agentic",
    },
    "minilm_wands": {"builder": make_wands_embedding_tool, "kind": "agentic"},
    "e5_base_v2_wands": {
        "builder": lambda corpus, device=None, **kwargs: make_wands_embedding_tool(
            corpus,
            device=device,
            model_name="intfloat/e5-base-v2",
            query_prefix="query: ",
            document_prefix="passage: ",
            **kwargs,
        ),
        "kind": "agentic",
    },
    "e5_base_v2_wands_prefiltered": {
        "builder": lambda corpus, device=None, **kwargs: make_wands_embedding_prefiltered_tool(
            corpus,
            device=device,
            model_name="intfloat/e5-base-v2",
            query_prefix="query: ",
            document_prefix="passage: ",
            **kwargs,
        ),
        "kind": "agentic",
    },
    "top_categories": {"builder": make_top_categories_tool, "kind": "agentic"},
    "check_features_wands": {"builder": make_check_features_wands_tool, "kind": "agentic"},
}


def tool_kind(name: str) -> str:
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown search tool: {name}")
    return entry.get("kind", "agentic")
