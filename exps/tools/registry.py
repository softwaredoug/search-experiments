from __future__ import annotations

from exps.tools.bm25 import make_bm25_tool, make_fielded_bm25_tool
from exps.tools.codegen import make_codegen_tool
from exps.tools.embeddings import make_embedding_tool
from exps.tools.query_rewrite import make_query_rewrite_tool
from exps.tools.raw import make_get_corpus_tool
from exps.tools.wands import (
    make_check_features_wands_tool,
    make_wands_bm25_tool,
    make_wands_embedding_tool,
)

TOOL_REGISTRY = {
    "bm25": {"builder": make_bm25_tool, "kind": "agentic"},
    "fielded_bm25": {"builder": make_fielded_bm25_tool, "kind": "agentic"},
    "minilm": {"builder": make_embedding_tool, "kind": "agentic"},
    "embeddings": {"builder": make_embedding_tool, "kind": "agentic"},
    "codegen": {"builder": make_codegen_tool, "kind": "agentic"},
    "query_rewrite": {"builder": make_query_rewrite_tool, "kind": "agentic"},
    "get_corpus": {"builder": make_get_corpus_tool, "kind": "raw"},
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
    "minilm_wands": {"builder": make_wands_embedding_tool, "kind": "agentic"},
    "e5_base_v2_wands": {
        "builder": lambda corpus, device=None: make_wands_embedding_tool(
            corpus,
            device=device,
            model_name="intfloat/e5-base-v2",
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
        "kind": "agentic",
    },
    "check_features_wands": {"builder": make_check_features_wands_tool, "kind": "agentic"},
}


def tool_kind(name: str) -> str:
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        raise ValueError(f"Unknown search tool: {name}")
    return entry.get("kind", "agentic")
