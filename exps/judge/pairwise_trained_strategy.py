from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import pandas as pd
from cheat_at_search.strategy import SearchStrategy
import joblib

from exps.judge.pairwise_train import (
    PairwiseFeatureComputer,
    _predict_pairwise_outcomes,
    _sample_documents,
    train_pairwise_strategy,
)
from exps.train_runs import find_latest_train_run


class PairwiseTrainedJudgeStrategy(SearchStrategy):
    """Rank sampled judged documents using a trained binary pairwise model."""

    _type = "pairwise_trained_judge"

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
        del device
        return train_pairwise_strategy(
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
        judgments,
        workers: int = 1,
        dataset: str | None = None,
        strategy_name: str | None = None,
        run_path: str | Path | None = None,
        no_cache: bool = False,
        **_kwargs,
    ):
        if judgments is None:
            raise ValueError("The pairwise-trained judge requires dataset judgments.")
        if run_path is None:
            if dataset is None or strategy_name is None:
                raise ValueError(
                    "Pairwise-trained judge requires dataset and strategy name to locate its artifact."
                )
            run_path = find_latest_train_run(
                dataset=dataset,
                strategy_name=strategy_name,
                strategy_type=cls._type,
            )
            if run_path is None:
                raise ValueError(
                    "No trained pairwise judge found. Run `uv run train` or set strategy.path."
                )
        run_path = Path(run_path).expanduser()
        if not run_path.exists() or not run_path.is_dir():
            raise FileNotFoundError(f"Pairwise-trained judge artifact not found: {run_path}")
        model_path = run_path / "model.joblib"
        schema_path = run_path / "feature_schema.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Pairwise-trained judge model not found: {model_path}")
        if not schema_path.exists():
            raise FileNotFoundError(f"Pairwise-trained judge feature schema not found: {schema_path}")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        return cls(
            corpus,
            judgments=judgments,
            model=joblib.load(model_path),
            feature_schema=schema,
            artifact_path=run_path,
            workers=workers,
            no_cache=no_cache,
        )

    def __init__(
        self,
        corpus,
        *,
        judgments: pd.DataFrame,
        model,
        feature_schema: dict,
        artifact_path: str | Path,
        workers: int = 1,
        no_cache: bool = False,
    ):
        self.corpus = corpus
        self.model = model
        self.artifact_path = Path(artifact_path)
        signature = hashlib.sha256()
        signature.update((self.artifact_path / "model.joblib").read_bytes())
        signature.update((self.artifact_path / "feature_schema.json").read_bytes())
        self.model_signature = signature.hexdigest()
        self.sample_size = feature_schema["sample_size"]
        self.sampling_seed = feature_schema["sampling_seed"]
        model_config = feature_schema.get("model", {})
        self.prediction_threshold = (
            model_config.get("params", {}).get("prediction_threshold")
            if isinstance(model_config, dict)
            else None
        )
        if feature_schema.get("to_pointwise") != "copeland":
            raise ValueError("Pairwise-trained judge artifact must use Copeland conversion.")
        self.feature_names = [
            next(iter(feature)) for feature in feature_schema["features"]
        ]
        self.feature_computer = PairwiseFeatureComputer(
            corpus=corpus,
            features=feature_schema["features"],
            no_cache=no_cache,
        )
        self.samples = _sample_documents(
            corpus=corpus,
            judgments=judgments,
            sample_size=self.sample_size,
            seed=self.sampling_seed,
        )
        super().__init__(corpus, workers=workers)

    def search(self, query, k: int = 10):
        sampled = self.samples.get(str(query), [])[:k]
        points = {index: 0.0 for index, _ in sampled}
        if len(sampled) < 2:
            ranked = sorted(points.items())
            return [index for index, _ in ranked], [score for _, score in ranked]
        pairs = list(itertools.combinations(sampled, 2))
        feature_rows = []
        for (lhs_index, _), (rhs_index, _) in pairs:
            feature_rows.append(
                self.feature_computer.compute(
                    str(query),
                    self.corpus.iloc[lhs_index],
                    self.corpus.iloc[rhs_index],
                )
            )
        predictions = _predict_pairwise_outcomes(
            self.model,
            pd.DataFrame(feature_rows, columns=self.feature_names),
            self.prediction_threshold,
        )
        for ((lhs_index, _), (rhs_index, _)), winner in zip(pairs, predictions):
            if winner == "lhs":
                points[lhs_index] += 1.0
            elif winner == "rhs":
                points[rhs_index] += 1.0
            elif winner == "neither":
                points[lhs_index] += 0.5
                points[rhs_index] += 0.5
            else:
                raise ValueError(f"Pairwise model returned invalid class {winner!r}.")
        ranked = sorted(points.items(), key=lambda item: (-item[1], item[0]))
        return [index for index, _ in ranked], [score for _, score in ranked]

    @property
    def cache_key(self) -> str:
        digest = hashlib.sha256(self.model_signature.encode("utf-8"))
        digest.update(json.dumps(self.samples, sort_keys=True).encode("utf-8"))
        return f"{self._type}:{digest.hexdigest()}"
