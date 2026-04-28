from __future__ import annotations

from pathlib import Path

import numpy as np
from cheat_at_search.strategy import SearchStrategy

from exps.codegen.utils import build_id_lookup, load_rerank_fn, resolve_id_column


class CodeGenSearchStrategy(SearchStrategy):
    _type = "codegen"

    def __init__(
        self,
        corpus,
        *,
        search_fn,
        code: str,
        rerank_name: str | None = None,
        artifact_path: Path | None = None,
        workers: int = 1,
    ):
        super().__init__(corpus, workers=workers)
        self.index = corpus
        self.search_fn = search_fn
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
        from exps.codegen.train import train_codegen_strategy

        artifact = train_codegen_strategy(
            strategy_name=strategy_name,
            dataset=dataset,
            corpus=corpus,
            judgments=kwargs.get("judgments"),
            params=params,
            device=device,
            workers=workers,
            report_num_queries=report_num_queries,
            report_seed=report_seed or 42,
        )
        rerank_name = f"rerank_{dataset}"
        return cls(
            corpus,
            workers=workers,
            search_fn=artifact.search_fn,
            code=artifact.code,
            rerank_name=rerank_name,
            artifact_path=artifact.path,
        )

    def search(self, query, k: int = 10):
        rerank_fn = load_rerank_fn(self.code, self.rerank_name)
        doc_ids = rerank_fn(self.search_fn, query)[:k]
        scores = np.arange(len(doc_ids), 0, -1)
        top_k_ilocs = []
        for doc_id in doc_ids:
            iloc = self._lookup.get(str(doc_id))
            if iloc is not None:
                top_k_ilocs.append(iloc)
        scores = scores[: len(top_k_ilocs)]
        return top_k_ilocs, scores
