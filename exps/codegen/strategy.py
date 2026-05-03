from __future__ import annotations

from pathlib import Path

import numpy as np
from cheat_at_search.strategy import SearchStrategy

from exps.codegen.io import find_latest_codegen_run, reranker_path
from exps.codegen.utils import build_id_lookup, load_rerank_fn, resolve_id_column, split_search_tools
from exps.tools import build_search_tools


class CodeGenSearchStrategy(SearchStrategy):
    _type = "codegen"

    def __init__(
        self,
        corpus,
        *,
        search_fn,
        tool_fns: list[callable] | None = None,
        code: str,
        rerank_name: str | None = None,
        artifact_path: Path | None = None,
        workers: int = 1,
    ):
        super().__init__(corpus, workers=workers)
        self.index = corpus
        self.search_fn = search_fn
        self.tool_fns = tool_fns or [search_fn]
        self.code = code
        self.rerank_name = rerank_name
        self.artifact_path = artifact_path
        id_col = resolve_id_column(corpus)
        self._lookup = build_id_lookup(corpus, id_col)

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        device: str | None = None,
        dataset: str | None = None,
        strategy_name: str | None = None,
        report_num_queries: int | None = None,
        report_seed: int | None = None,
        **kwargs,
    ):
        if dataset is None:
            raise ValueError("Codegen strategy requires dataset name.")
        if strategy_name is None:
            raise ValueError("Codegen strategy requires strategy name.")
        run_config = dict(params.get("run") or {})
        run_path = run_config.get("path")
        if run_path is None:
            run_path = find_latest_codegen_run(dataset, strategy_name)
            if run_path is None:
                raise ValueError(
                    "No trained codegen run found. Run `uv run train` or set run.path."
                )
        run_path = Path(run_path).expanduser()
        if not run_path.exists():
            raise FileNotFoundError(f"Codegen run path not found: {run_path}")

        code_path = reranker_path(run_path)
        if not code_path.exists():
            raise FileNotFoundError(f"Reranker code not found: {code_path}")
        code = code_path.read_text(encoding="utf-8")

        train_config = dict(params.get("train") or {})
        tool_config = train_config.get("search_tools") or []
        normal_tool_config, raw_tool_config = split_search_tools(tool_config)
        tool_fns = build_search_tools(
            corpus,
            normal_tool_config,
            embeddings_device=device,
            dataset_name=dataset,
        )
        raw_tools = build_search_tools(
            corpus,
            raw_tool_config,
            embeddings_device=device,
            dataset_name=dataset,
            context="raw",
        )
        if not tool_fns and not raw_tools:
            raise ValueError("Codegen run requires at least one search tool.")
        tool_fns = tool_fns + raw_tools
        if not tool_fns:
            raise ValueError("Codegen run requires at least one search tool.")
        rerank_name = f"rerank_{dataset}"
        return cls(
            corpus,
            workers=workers,
            search_fn=tool_fns[0],
            tool_fns=tool_fns,
            code=code,
            rerank_name=rerank_name,
            artifact_path=run_path,
        )

    def search(self, query, k: int = 10):
        try:
            rerank_fn = load_rerank_fn(self.code, self.rerank_name)
            doc_ids = rerank_fn(query, *self.tool_fns)[:k]
            scores = np.arange(len(doc_ids), 0, -1)
            top_k_ilocs = []
            for doc_id in doc_ids:
                iloc = self._lookup.get(str(doc_id))
                if iloc is not None:
                    top_k_ilocs.append(iloc)
            scores = scores[: len(top_k_ilocs)]
            return top_k_ilocs, scores
        except Exception:
            return [], np.array([])
