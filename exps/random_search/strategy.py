from __future__ import annotations

import json
from pathlib import Path

from cheat_at_search.strategy import SearchStrategy

from exps.random_search.train import train_random_search_strategy
from exps.strategies.bm25 import BM25Strategy
from exps.train_runs import find_latest_train_run


class RandomSearchStrategy(SearchStrategy):
    """Run the BM25 configuration selected by a random-search training run."""

    _type = "random_search"

    def __init__(self, corpus, *, engine: BM25Strategy, artifact_path: Path, workers: int = 1):
        super().__init__(corpus, workers=workers)
        self.engine = engine
        self.artifact_path = artifact_path

    @classmethod
    def train(
        cls,
        *,
        strategy_name: str,
        dataset: str,
        corpus,
        judgments,
        params: dict,
        run_path: str | Path,
        device: str | None = None,
        workers: int = 1,
        report_num_queries: int | None = None,
        report_seed: int = 42,
    ):
        del device  # BM25 training does not use an embedding device.
        return train_random_search_strategy(
            strategy_name=strategy_name,
            dataset=dataset,
            corpus=corpus,
            judgments=judgments,
            params=params,
            run_path=run_path,
            workers=workers,
            report_num_queries=report_num_queries,
            report_seed=report_seed,
        )

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        dataset: str | None = None,
        strategy_name: str | None = None,
        run_path: str | Path | None = None,
        **_kwargs,
    ):
        if dataset is None:
            raise ValueError("Random-search strategy requires dataset name.")
        if strategy_name is None:
            raise ValueError("Random-search strategy requires strategy name.")
        if run_path is None:
            run_path = find_latest_train_run(
                dataset=dataset,
                strategy_name=strategy_name,
                strategy_type=cls._type,
            )
            if run_path is None:
                raise ValueError(
                    "No trained random-search run found. Run `uv run train` or set strategy.path."
                )
        run_path = Path(run_path).expanduser()
        if not run_path.exists() or not run_path.is_dir():
            raise FileNotFoundError(f"Random-search run path not found: {run_path}")
        best_params_path = run_path / "best_params.json"
        if not best_params_path.exists():
            raise FileNotFoundError(f"Random-search artifact not found: {best_params_path}")
        artifact = json.loads(best_params_path.read_text(encoding="utf-8"))
        engine_config = artifact.get("retrieval_engine") or {}
        if engine_config.get("base") != "bm25":
            raise ValueError("Random-search artifact must contain a BM25 retrieval engine.")
        engine_params = engine_config.get("params")
        if not isinstance(engine_params, dict):
            raise ValueError("Random-search artifact has invalid BM25 params.")
        engine = BM25Strategy.build(engine_params, corpus=corpus, workers=workers)
        return cls(corpus, engine=engine, artifact_path=run_path, workers=workers)

    def search(self, query, k: int = 10):
        return self.engine.search(query, k=k)

    @property
    def cache_key(self) -> str:
        return f"{self._type}:{self.engine.cache_key}"
