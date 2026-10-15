from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import joblib
import pytest

from exps.runners.run import RunParams, run_benchmark
from exps.runners.train import TrainParams, train_strategy


class ScriptedPairwiseFeatureEnricher:
    instances = []
    active_calls = 0
    max_active_calls = 0
    calls_lock = threading.Lock()

    def __init__(self, *, model, system_prompt, response_model):
        self.model = model
        self.system_prompt = system_prompt
        self.response_model = response_model
        self.prompts = []
        self.enricher = self
        self.cached_enricher = self
        self.cache_file = "scripted-openai-cache"
        self.cache = {}
        type(self).instances.append(self)

    def enrich(self, prompt):
        self.prompts.append(prompt)
        with type(self).calls_lock:
            type(self).active_calls += 1
            type(self).max_active_calls = max(
                type(self).max_active_calls, type(self).active_calls
            )
        try:
            time.sleep(0.01)
            lhs_grade, rhs_grade = map(
                int,
                re.search(
                    r"LHS (?:title|description): grade=(\d+)\s+"
                    r"RHS (?:title|description): grade=(\d+)",
                    prompt,
                ).groups(),
            )
            winner = "lhs" if lhs_grade > rhs_grade else "rhs"
            return self.response_model(winner=winner)
        finally:
            with type(self).calls_lock:
                type(self).active_calls -= 1

    def save_cache(self):
        pass


class ScriptedPairwiseDecisionEnricher:
    instances = []
    active_calls = 0
    max_active_calls = 0
    calls_lock = threading.Lock()

    def __init__(self, *, choices, model):
        self.choices = choices
        self.model = model
        self.calls = []
        self.delete_cache_calls = 0
        self.save_cache_calls = 0
        self.cached_enricher = self
        type(self).instances.append(self)

    def score(self, *, state, question):
        self.calls.append((state, question))
        with type(self).calls_lock:
            type(self).active_calls += 1
            type(self).max_active_calls = max(
                type(self).max_active_calls, type(self).active_calls
            )
        try:
            time.sleep(0.01)
            lhs_grade, rhs_grade = map(
                int,
                re.search(
                    r"LHS title: grade=(\d+)\s+RHS title: grade=(\d+)",
                    state,
                ).groups(),
            )
            winner = "lhs" if lhs_grade > rhs_grade else "rhs"
            loser = "rhs" if winner == "lhs" else "lhs"
            return {winner: 0.99, loser: 0.01}, 0.95
        finally:
            with type(self).calls_lock:
                type(self).active_calls -= 1

    def delete_cache(self):
        self.delete_cache_calls += 1

    def save_cache(self):
        self.save_cache_calls += 1


def _dataset():
    queries = [f"query-{index}" for index in range(6)]
    corpus_rows = []
    judgment_rows = []
    doc_id = 0
    for query_id, query in enumerate(queries):
        for grade in (2, 1, 0):
            corpus_rows.append(
                {
                    "doc_id": doc_id,
                    "title": f"grade={grade}",
                    "description": f"grade={grade}",
                }
            )
            judgment_rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "doc_id": doc_id,
                    "grade": grade,
                }
            )
            doc_id += 1
    return SimpleNamespace(
        corpus=pd.DataFrame(corpus_rows),
        judgments=pd.DataFrame(judgment_rows),
    )


def _write_config(path, *, artifact_path=None, gradient_boosted=False):
    strategy_path = "" if artifact_path is None else f"  path: {artifact_path}\n"
    model_config = (
        "      type: gradient_boosted_tree\n"
        "      params:\n"
        "        n_estimators: 5\n"
        "        max_depth: 1\n"
        "        prediction_threshold: 0.7\n"
        if gradient_boosted
        else "      type: decision_tree\n"
        "      params:\n"
        "        prediction_threshold: 0.7\n"
    )
    path.write_text(
        f"""
strategy:
  name: pairwise_trained_fixture
  type: pairwise_trained_judge
{strategy_path}  params:
    sample_size: 3
    sampling_seed: 17
    to_pointwise: copeland
    model:
{model_config}    features:
      - title:
          model: jev/jev-latest
          prompt: |
            Query: {{query}}
            LHS title: {{title_lhs}}
            RHS title: {{title_rhs}}
          question: Which title is more relevant?
      - description:
          model: gpt-5-mini
          system_prompt: Compare description relevance.
          prompt: |
            Query: {{query}}
            LHS description: {{description_lhs}}
            RHS description: {{description_rhs}}
          question: Which description is more relevant?
    train:
      cross_validation: 3
""".lstrip(),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("gradient_boosted", "expected_model"),
    [
        (
            False,
            {"type": "decision_tree", "params": {"prediction_threshold": 0.7}},
        ),
        (
            True,
            {
                "type": "gradient_boosted_tree",
                "params": {
                    "n_estimators": 5,
                    "max_depth": 1,
                    "prediction_threshold": 0.7,
                },
            },
        ),
    ],
)
def test_pairwise_trained_judge_cross_validates_fits_artifact_and_runs(
    tmp_path, monkeypatch, capsys, gradient_boosted, expected_model
):
    from exps import paths, train_runs
    from exps.runners import run as run_module
    from exps.runners import train as train_module

    ScriptedPairwiseFeatureEnricher.instances = []
    ScriptedPairwiseFeatureEnricher.active_calls = 0
    ScriptedPairwiseFeatureEnricher.max_active_calls = 0
    ScriptedPairwiseDecisionEnricher.instances = []
    ScriptedPairwiseDecisionEnricher.active_calls = 0
    ScriptedPairwiseDecisionEnricher.max_active_calls = 0
    dataset = _dataset()
    artifact_root = tmp_path / "artifacts"
    monkeypatch.setattr(paths, "SEARCH_EXPERIMENTS_ROOT", artifact_root)
    monkeypatch.setattr(train_runs, "SEARCH_EXPERIMENTS_ROOT", artifact_root)
    monkeypatch.setattr(train_module, "get_dataset", lambda *args, **kwargs: dataset)
    monkeypatch.setattr(run_module, "get_dataset", lambda *args, **kwargs: dataset)

    with patch(
        "exps.judge.engines.llm_pairwise.AutoEnricher",
        ScriptedPairwiseFeatureEnricher,
    ), patch(
        "exps.judge.engines.jev_pairwise.AutoDecisionEnricher",
        ScriptedPairwiseDecisionEnricher,
    ):
        config_path = _write_config(
            tmp_path / "trained_judge.yml", gradient_boosted=gradient_boosted
        )
        trained = train_strategy(
            TrainParams(
                strategy_path=str(config_path),
                dataset="doug_blog",
                seed=17,
                workers=3,
                device="cpu",
            )
        )
        progress_output = capsys.readouterr().err
        assert "Generating pairwise features" in progress_output
        assert "6/6" in progress_output

        artifact_path = Path(trained.artifact_path)
        assert (artifact_path / "model.joblib").exists()
        assert (artifact_path / "metadata.json").exists()
        assert trained.metadata["cross_validation"]["folds"] == 3
        assert len(trained.metadata["cross_validation"]["fold_scores"]) == 3
        assert trained.metadata["cross_validation"]["mean_precision"] == 1.0
        assert trained.metadata["cross_validation"]["mean_prediction_coverage"] > 0.5
        if not gradient_boosted:
            assert trained.metadata["cross_validation"]["mean_prediction_coverage"] == 1.0
        for feature_name in ("title", "description"):
            feature_metrics = trained.metadata["cross_validation"]["feature_metrics"][
                feature_name
            ]
            assert feature_metrics["mean_accuracy"] == 1.0
            assert feature_metrics["mean_precision"] == 1.0
            assert feature_metrics["mean_prediction_coverage"] == 1.0
            assert len(feature_metrics["folds_detail"]) == 3
        assert trained.metadata["training_pair_count"] == 18
        assert trained.metadata["model"] == expected_model
        assert set(trained.metadata["target_classes"]) == {"rhs", "lhs"}
        fold_details = trained.metadata["cross_validation"]["folds_detail"]
        validation_queries = []
        for fold in fold_details:
            assert set(fold["train_queries"]).isdisjoint(fold["validation_queries"])
            validation_queries.extend(fold["validation_queries"])
        assert sorted(validation_queries) == [f"query-{index}" for index in range(6)]
        trained_model = joblib.load(artifact_path / "model.joblib")
        if gradient_boosted:
            estimator = trained_model.named_steps["gradientboostingclassifier"]
            assert estimator.n_estimators == 5
            assert estimator.max_depth == 1
        else:
            assert (
                trained_model.named_steps["decisiontreeclassifier"]
                .tree_.n_node_samples[0]
                == 18
            )

        runtime_config = _write_config(
            tmp_path / "trained_judge_run.yml",
            artifact_path=artifact_path,
            gradient_boosted=gradient_boosted,
        )
        result = run_benchmark(
            RunParams(
                strategy_path=str(runtime_config),
                dataset="doug_blog",
                workers=1,
                seed=17,
                device="cpu",
                no_cache=True,
            )
        )

    assert result.artifact_path == str(artifact_path)
    assert result.summary["mean_resource.model_calls"] == 12.0
    if gradient_boosted:
        assert result.summary["mean_pairwise_agreement"] > 0.8
    else:
        assert result.summary["mean_pairwise_agreement"] == 1.0
    assert len(ScriptedPairwiseFeatureEnricher.instances) == 2
    assert all(instance.prompts for instance in ScriptedPairwiseFeatureEnricher.instances)
    assert ScriptedPairwiseFeatureEnricher.max_active_calls > 1
    assert len(ScriptedPairwiseDecisionEnricher.instances) == 2
    assert [instance.delete_cache_calls for instance in ScriptedPairwiseDecisionEnricher.instances] == [0, 1]
    assert [instance.save_cache_calls for instance in ScriptedPairwiseDecisionEnricher.instances] == [1, 0]
    assert ScriptedPairwiseDecisionEnricher.max_active_calls > 1
