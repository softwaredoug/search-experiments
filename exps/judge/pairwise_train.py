from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import string
from collections.abc import Sized
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

from exps.judge.classifiers import build_classifier, normalize_model_config
from exps.judge.engines.common import PairwiseJudgeResult, grade_value
from exps.judge.engines.jev_pairwise import JevPairwiseJudgeEngine
from exps.judge.engines.llm_pairwise import LLMPairwiseJudgeEngine


_FEATURE_OPTIONS = {
    "model",
    "system_prompt",
    "prompt",
    "question",
    "confidence_threshold",
    "explicit_neither",
    "double_check",
    "empty_value",
}
_GRADE_COLUMNS = ("grade", "relevance", "rel", "label", "score")


def _template_fields(template: str) -> set[str]:
    try:
        return {
            field_name.split(".", 1)[0].split("[", 1)[0]
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        }
    except ValueError as exc:
        raise ValueError("Pairwise feature templates must be valid format strings.") from exc


def _normalize_features(features: Any, corpus: pd.DataFrame) -> list[dict[str, Any]]:
    if not isinstance(features, list) or not features:
        raise ValueError("strategy.params.features must be a non-empty list.")
    normalized = []
    names = set()
    pair_fields = {
        f"{column}_{side}" for column in corpus.columns for side in ("lhs", "rhs")
    }
    for entry in features:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError("Each pairwise feature must map one feature name to its config.")
        name, raw_config = next(iter(entry.items()))
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Pairwise feature names must be non-empty and unique.")
        if not isinstance(raw_config, dict):
            raise ValueError(f"Feature {name!r} must have a mapping configuration.")
        unknown = set(raw_config) - _FEATURE_OPTIONS
        if unknown:
            raise ValueError(
                f"Unsupported option(s) for feature {name!r}: "
                + ", ".join(sorted(unknown))
            )
        config = dict(raw_config)
        for required in ("model", "prompt", "question"):
            if not isinstance(config.get(required), str) or not config[required].strip():
                raise ValueError(f"Feature {name!r} requires a non-empty {required!r}.")
        config.setdefault("system_prompt", "You are a careful search relevance judge.")
        config.setdefault("explicit_neither", False)
        config.setdefault("double_check", True)
        config.setdefault("empty_value", "Unknown")
        if not isinstance(config["system_prompt"], str) or not config["system_prompt"].strip():
            raise ValueError(f"Feature {name!r} system_prompt must be a non-empty string.")
        if not isinstance(config["explicit_neither"], bool):
            raise ValueError(f"Feature {name!r} explicit_neither must be a boolean.")
        if not isinstance(config["double_check"], bool):
            raise ValueError(f"Feature {name!r} double_check must be a boolean.")
        if (
            not isinstance(config["empty_value"], str)
            or not config["empty_value"].strip()
        ):
            raise ValueError(f"Feature {name!r} empty_value must be a non-empty string.")
        threshold = config.get("confidence_threshold")
        if threshold is not None:
            if (
                isinstance(threshold, bool)
                or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)
                or not 0 <= threshold <= 1
            ):
                raise ValueError(
                    f"Feature {name!r} confidence_threshold must be between zero and one."
                )
        model = config["model"].strip()
        provider, separator, model_name = model.partition("/")
        if separator and (provider.lower() not in {"jev", "openai"} or not model_name.strip()):
            raise ValueError(
                f"Feature {name!r} model must be an OpenAI model or a Jev model (jev/*)."
            )
        is_jev = provider.lower() == "jev"
        if threshold is not None and not is_jev:
            raise ValueError(
                f"Feature {name!r} confidence_threshold requires a Jev model."
            )
        allowed_fields = pair_fields | {"query", "question"}
        missing_fields = sorted(
            (_template_fields(config["prompt"]) - allowed_fields)
            | (_template_fields(config["question"]) - (pair_fields | {"query"}))
        )
        if missing_fields:
            message = (
                f"Feature {name!r} has unsupported prompt placeholder(s): "
                + ", ".join(missing_fields)
            )
            bare_corpus_fields = sorted(set(missing_fields) & set(corpus.columns))
            if bare_corpus_fields:
                examples = ", ".join(
                    f"{{{field}_lhs}} and {{{field}_rhs}}"
                    for field in bare_corpus_fields
                )
                message += (
                    ". In pairwise prompts, corpus columns must be suffixed with "
                    f"'_lhs' or '_rhs' (for example, {examples})."
                )
            raise ValueError(message)
        config["model"] = model
        config["is_jev"] = is_jev
        config["name"] = name
        names.add(name)
        normalized.append(config)
    return normalized


def _replace_empty_prompt_value(value: Any, empty_value: str) -> Any:
    if isinstance(value, str) and not value.strip():
        return empty_value
    try:
        is_missing = pd.isna(value)
        if not hasattr(is_missing, "__len__") and bool(is_missing):
            return empty_value
    except (TypeError, ValueError):
        pass
    try:
        if isinstance(value, Sized) and len(value) == 0:
            return empty_value
    except (TypeError, ValueError):
        pass
    return value


class PairwiseFeatureComputer:
    """Compute categorical pairwise-judge outputs for configured features."""

    def __init__(self, *, corpus: pd.DataFrame, features: list[dict], no_cache: bool = False):
        self.corpus = corpus
        self.features = _normalize_features(features, corpus)
        self.no_cache = no_cache
        self.engines = []
        shared_engines = {}
        for config in self.features:
            if config["is_jev"]:
                _, separator, model_name = config["model"].partition("/")
                engine_key = (
                    "jev",
                    (model_name if separator else "jev-latest").lower(),
                    config["explicit_neither"],
                )
                if engine_key not in shared_engines:
                    shared_engines[engine_key] = JevPairwiseJudgeEngine(
                        model=config["model"],
                        explicit_neither=config["explicit_neither"],
                        no_cache=no_cache,
                    )
            else:
                provider, separator, model_name = config["model"].partition("/")
                engine_key = (
                    "openai",
                    (model_name if separator else provider).lower(),
                    config["system_prompt"],
                    config["explicit_neither"],
                )
                if engine_key not in shared_engines:
                    shared_engines[engine_key] = LLMPairwiseJudgeEngine(
                        model=config["model"],
                        system_prompt=config["system_prompt"],
                        explicit_neither=config["explicit_neither"],
                        no_cache=no_cache,
                    )
            self.engines.append(shared_engines[engine_key])

    def _engine_for(self, index: int, _config: dict):
        return self.engines[index]

    def _judge(self, config: dict, engine, query: str, lhs, rhs) -> PairwiseJudgeResult:
        empty_value = config["empty_value"]
        values = {"query": _replace_empty_prompt_value(query, empty_value)}
        values.update(
            {
                f"{key}_lhs": _replace_empty_prompt_value(value, empty_value)
                for key, value in lhs.items()
            }
        )
        values.update(
            {
                f"{key}_rhs": _replace_empty_prompt_value(value, empty_value)
                for key, value in rhs.items()
            }
        )
        try:
            question = config["question"].format_map(values)
            values["question"] = question
            prompt = config["prompt"].format_map(values)
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(
                f"Could not render prompt for pairwise feature {config['name']!r}."
            ) from exc
        if config["is_jev"]:
            result = engine.judge(state=prompt, question=question)
        else:
            result = engine.judge(
                prompt=prompt,
                question=question,
                prompt_has_question="question" in _template_fields(config["prompt"]),
            )
        outcome = getattr(result, "outcome", result)
        if outcome not in {"lhs", "rhs", "neither"}:
            raise ValueError(f"Feature {config['name']!r} returned invalid outcome {outcome!r}.")
        probability = getattr(result, "probability", None)
        threshold = config.get("confidence_threshold")
        if threshold is not None and (probability is None or probability <= threshold):
            return PairwiseJudgeResult(outcome="neither")
        return PairwiseJudgeResult(outcome=outcome, probability=probability)

    def compute(self, query: str, lhs, rhs) -> dict[str, str]:
        output = {}
        for index, config in enumerate(self.features):
            engine = self._engine_for(index, config)
            first = self._judge(config, engine, query, lhs, rhs).outcome
            if config["double_check"]:
                swapped = self._judge(config, engine, query, rhs, lhs).outcome
                reverse = {"lhs": "rhs", "rhs": "lhs", "neither": "neither"}
                if reverse[first] != swapped:
                    first = "neither"
            output[config["name"]] = first
        return output

    def flush_caches(self) -> None:
        flushed = set()
        for engine in self.engines:
            if id(engine) in flushed:
                continue
            engine.flush_cache()
            flushed.add(id(engine))


def _sample_documents(
    *, corpus: pd.DataFrame, judgments: pd.DataFrame, sample_size: int, seed: int
) -> dict[str, list[tuple[int, float]]]:
    grade_column = next((column for column in _GRADE_COLUMNS if column in judgments), None)
    if grade_column is None:
        raise ValueError("Judgments must contain a numeric grade column.")
    if "query" not in judgments or "doc_id" not in judgments:
        raise ValueError("Judgments must contain 'query' and 'doc_id' columns.")
    if "doc_id" not in corpus:
        raise ValueError("The corpus must contain a 'doc_id' column.")
    doc_indices = {
        doc_id: index
        for index, doc_id in enumerate(corpus["doc_id"].tolist())
        if not pd.isna(doc_id)
    }
    samples = {}
    available = judgments.dropna(subset=["query", "doc_id", grade_column])
    for query, query_rows in available.groupby("query", sort=False):
        by_grade: dict[float, list[tuple[int, float]]] = {}
        for _, row in query_rows.iterrows():
            index = doc_indices.get(row["doc_id"])
            if index is None:
                continue
            grade = grade_value(row[grade_column])
            by_grade.setdefault(grade, []).append((index, grade))
        grades = sorted(by_grade)
        digest = hashlib.sha256(f"{seed}\0{query}".encode("utf-8")).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        for rows in by_grade.values():
            rng.shuffle(rows)
        target = min(sample_size, sum(map(len, by_grade.values())))
        start_grade = rng.randrange(len(grades)) if grades else 0
        allocation_order = grades[start_grade:] + grades[:start_grade]
        quotas = {
            grade: min(target // len(grades), len(by_grade[grade]))
            for grade in grades
        } if grades else {}
        remaining = target - sum(quotas.values())
        while remaining:
            progressed = False
            for grade in allocation_order:
                if quotas[grade] < len(by_grade[grade]):
                    quotas[grade] += 1
                    remaining -= 1
                    progressed = True
                    if remaining == 0:
                        break
            if not progressed:
                break
        selected = []
        for grade in grades:
            selected.extend(by_grade[grade][: quotas[grade]])
        rng.shuffle(selected)
        samples[str(query)] = selected
    return samples


def build_binary_training_data(
    comparisons: pd.DataFrame, *, feature_columns: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return feature rows, binary higher-grade-side targets, and query groups."""
    required = {"query", "grade_lhs", "grade_rhs", *feature_columns}
    missing = required - set(comparisons.columns)
    if missing:
        raise ValueError("Pairwise training rows are missing: " + ", ".join(sorted(missing)))
    retained = []
    targets = []
    groups = []
    for _, row in comparisons.iterrows():
        lhs_grade = grade_value(row["grade_lhs"])
        rhs_grade = grade_value(row["grade_rhs"])
        if lhs_grade == rhs_grade:
            continue
        retained.append({column: row[column] for column in feature_columns})
        targets.append("lhs" if lhs_grade > rhs_grade else "rhs")
        groups.append(row["query"])
    if not retained:
        raise ValueError("Pairwise training requires at least one unequal-grade document pair.")
    return (
        pd.DataFrame(retained, columns=feature_columns),
        pd.Series(targets, name="target"),
        pd.Series(groups, name="query"),
    )


def query_group_folds(groups, *, n_splits: int):
    unique_groups = pd.Series(groups).nunique()
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("train.cross_validation must be an integer of at least 2.")
    if n_splits > unique_groups:
        raise ValueError(
            "train.cross_validation cannot exceed the number of distinct training queries."
        )
    splitter = GroupKFold(n_splits=n_splits)
    import numpy as np

    indices = np.arange(len(groups))
    return list(splitter.split(indices, groups=groups))


def _new_pipeline(model_config: dict[str, Any]):
    return make_pipeline(
        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        build_classifier(model_config),
    )


def _normalize_model_config(raw_model: Any) -> dict[str, Any]:
    return normalize_model_config(raw_model)


def _predict_pairwise_outcomes(model, features, prediction_threshold: float | None):
    predictions = model.predict(features)
    if prediction_threshold is None:
        return predictions
    probabilities = model.predict_proba(features)
    return [
        prediction if max(class_probabilities) >= prediction_threshold else "neither"
        for prediction, class_probabilities in zip(predictions, probabilities)
    ]


def _pairwise_prediction_metrics(targets, predictions) -> dict[str, Any]:
    target_values = list(targets)
    prediction_values = list(predictions)
    if len(target_values) != len(prediction_values):
        raise ValueError("Pairwise targets and predictions must have equal lengths.")
    if not target_values:
        raise ValueError("Pairwise prediction metrics require at least one pair.")
    decided = [
        (target, prediction)
        for target, prediction in zip(target_values, prediction_values)
        if prediction in {"lhs", "rhs"}
    ]
    correct_decided = sum(target == prediction for target, prediction in decided)
    return {
        "accuracy": sum(
            target == prediction
            for target, prediction in zip(target_values, prediction_values)
        )
        / len(target_values),
        "precision": correct_decided / len(decided) if decided else None,
        "prediction_coverage": len(decided) / len(target_values),
        "decided_pair_count": len(decided),
        "correct_decided_pair_count": correct_decided,
    }


def train_pairwise_strategy(
    *,
    strategy_name: str,
    dataset: str,
    corpus: pd.DataFrame,
    judgments: pd.DataFrame,
    params: dict,
    run_path: str | Path,
    workers: int = 1,
    report_num_queries: int | None = None,
    report_seed: int = 42,
):
    run_path = Path(run_path).expanduser()
    if not run_path.exists() or not run_path.is_dir():
        raise ValueError(f"Pairwise training run path must be an existing directory: {run_path}")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer of at least one.")
    sample_size = params.get("sample_size", 10)
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or not 1 <= sample_size <= 10:
        raise ValueError("strategy.params.sample_size must be an integer from 1 to 10.")
    sampling_seed = params.get("sampling_seed", 42)
    if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
        raise ValueError("strategy.params.sampling_seed must be an integer.")
    if params.get("to_pointwise", "copeland") != "copeland":
        raise ValueError("strategy.params.to_pointwise must be 'copeland'.")
    model_config = _normalize_model_config(params.get("model"))
    prediction_threshold = model_config["params"].get("prediction_threshold")
    train_config = params.get("train") or {}
    if not isinstance(train_config, dict):
        raise ValueError("strategy.params.train must be a mapping.")
    cross_validation = train_config.get("cross_validation", 5)
    if set(train_config) != {"cross_validation"}:
        raise ValueError("strategy.params.train supports only 'cross_validation'.")
    if (
        isinstance(cross_validation, bool)
        or not isinstance(cross_validation, int)
        or cross_validation < 2
    ):
        raise ValueError("train.cross_validation must be an integer of at least 2.")
    if judgments is None or "query" not in judgments:
        raise ValueError("Pairwise training requires query judgments.")

    available_queries = judgments["query"].drop_duplicates().tolist()
    if report_num_queries is not None:
        if report_num_queries < 1:
            raise ValueError("num_queries must be at least one.")
        available_queries = random.Random(report_seed).sample(
            available_queries, min(report_num_queries, len(available_queries))
        )
    if not available_queries:
        raise ValueError("Pairwise training requires at least one judged query.")
    selected_judgments = judgments[judgments["query"].isin(available_queries)]
    samples = _sample_documents(
        corpus=corpus,
        judgments=selected_judgments,
        sample_size=sample_size,
        seed=sampling_seed,
    )
    trainable_query_count = sum(
        len({grade for _, grade in samples.get(str(query), [])}) > 1
        for query in available_queries
    )
    if cross_validation > trainable_query_count:
        raise ValueError(
            "train.cross_validation cannot exceed the number of queries with "
            "unequal-grade sampled pairs."
        )
    feature_computer = PairwiseFeatureComputer(corpus=corpus, features=params["features"])
    feature_configs = feature_computer.features
    grade_column = next(column for column in _GRADE_COLUMNS if column in selected_judgments)
    query_jobs = []
    for query in available_queries:
        docs = samples.get(str(query), [])
        query_pairs = []
        for (lhs_index, lhs_grade), (rhs_index, rhs_grade) in itertools.combinations(docs, 2):
            if lhs_grade == rhs_grade:
                continue
            query_pairs.append((lhs_index, lhs_grade, rhs_index, rhs_grade))
        if query_pairs:
            query_jobs.append((str(query), query_pairs))

    def compute_query(query_job):
        query, query_pairs = query_job
        rows = []
        for lhs_index, lhs_grade, rhs_index, rhs_grade in query_pairs:
            row = {
                "query": query,
                "grade_lhs": lhs_grade,
                "grade_rhs": rhs_grade,
            }
            row.update(
                feature_computer.compute(
                    query,
                    corpus.iloc[lhs_index],
                    corpus.iloc[rhs_index],
                )
            )
            rows.append(row)
        return rows

    comparisons_by_query = [None] * len(query_jobs)
    try:
        with tqdm(
            total=len(query_jobs),
            desc="Generating pairwise features",
            unit="query",
        ) as progress:
            if workers == 1:
                for index, query_job in enumerate(query_jobs):
                    comparisons_by_query[index] = compute_query(query_job)
                    progress.update(1)
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        executor.submit(compute_query, query_job): index
                        for index, query_job in enumerate(query_jobs)
                    }
                    for future in as_completed(futures):
                        comparisons_by_query[futures[future]] = future.result()
                        progress.update(1)
    finally:
        feature_computer.flush_caches()
    comparisons = [row for query_rows in comparisons_by_query for row in query_rows]

    feature_columns = [feature["name"] for feature in feature_configs]
    comparison_frame = pd.DataFrame(
        comparisons,
        columns=["query", "grade_lhs", "grade_rhs", *feature_columns],
    )
    features, targets, groups = build_binary_training_data(
        comparison_frame,
        feature_columns=feature_columns,
    )
    folds = query_group_folds(groups.tolist(), n_splits=cross_validation)
    fold_scores = []
    fold_precisions = []
    fold_metadata = []
    feature_fold_metrics: dict[str, list[dict[str, Any]]] = {
        feature_name: [] for feature_name in feature_columns
    }
    total_decided_pair_count = 0
    total_correct_decided_pair_count = 0
    total_validation_pair_count = 0
    for train_indices, validation_indices in folds:
        train_targets = targets.iloc[train_indices]
        if set(train_targets) != {"lhs", "rhs"}:
            raise ValueError(
                "Each cross-validation training fold must contain both pairwise outcomes; "
                "use more judged queries or fewer folds."
            )
        fold_model = _new_pipeline(model_config)
        fold_model.fit(features.iloc[train_indices], train_targets)
        predictions = _predict_pairwise_outcomes(
            fold_model,
            features.iloc[validation_indices],
            prediction_threshold,
        )
        metrics = _pairwise_prediction_metrics(
            targets.iloc[validation_indices], predictions
        )
        score = metrics["accuracy"]
        fold_scores.append(score)
        fold_precisions.append(metrics["precision"])
        total_decided_pair_count += metrics["decided_pair_count"]
        total_correct_decided_pair_count += metrics["correct_decided_pair_count"]
        total_validation_pair_count += len(validation_indices)
        per_feature_metrics = {}
        for feature_name in feature_columns:
            feature_metrics = _pairwise_prediction_metrics(
                targets.iloc[validation_indices],
                features.iloc[validation_indices][feature_name],
            )
            feature_metrics["validation_pair_count"] = len(validation_indices)
            feature_fold_metrics[feature_name].append(feature_metrics)
            per_feature_metrics[feature_name] = feature_metrics
        fold_metadata.append(
            {
                "accuracy": score,
                "precision": metrics["precision"],
                "prediction_coverage": metrics["prediction_coverage"],
                "decided_pair_count": metrics["decided_pair_count"],
                "train_queries": sorted(set(groups.iloc[train_indices].tolist())),
                "validation_queries": sorted(set(groups.iloc[validation_indices].tolist())),
                "train_pair_count": len(train_indices),
                "validation_pair_count": len(validation_indices),
                "feature_metrics": per_feature_metrics,
            }
        )

    cross_validation_feature_metrics = {}
    for feature_name, fold_metrics in feature_fold_metrics.items():
        decided_count = sum(metric["decided_pair_count"] for metric in fold_metrics)
        correct_decided_count = sum(
            metric["correct_decided_pair_count"] for metric in fold_metrics
        )
        validation_pair_count = sum(
            metric["validation_pair_count"] for metric in fold_metrics
        )
        cross_validation_feature_metrics[feature_name] = {
            "mean_accuracy": float(
                sum(metric["accuracy"] for metric in fold_metrics) / len(fold_metrics)
            ),
            "fold_accuracies": [metric["accuracy"] for metric in fold_metrics],
            "mean_precision": (
                correct_decided_count / decided_count if decided_count else None
            ),
            "fold_precisions": [metric["precision"] for metric in fold_metrics],
            "mean_prediction_coverage": decided_count / validation_pair_count,
            "folds_detail": fold_metrics,
        }

    if set(targets) != {"lhs", "rhs"}:
        raise ValueError("Pairwise training requires examples of both lhs and rhs wins.")
    final_model = _new_pipeline(model_config)
    final_model.fit(features, targets)
    joblib.dump(final_model, run_path / "model.joblib")
    schema = {
        "sample_size": sample_size,
        "sampling_seed": sampling_seed,
        "to_pointwise": "copeland",
        "model": model_config,
        "features": params["features"],
    }
    (run_path / "feature_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    metadata = {
        "strategy_type": "pairwise_trained_judge",
        "strategy_name": strategy_name,
        "dataset": dataset,
        "model": model_config,
        "target_classes": list(final_model.classes_),
        "feature_names": feature_columns,
        "base_queries": [str(query) for query in available_queries],
        "base_query_count": len(available_queries),
        "training_pair_count": len(features),
        "cross_validation": {
            "folds": cross_validation,
            "mean_accuracy": float(sum(fold_scores) / len(fold_scores)),
            "fold_scores": fold_scores,
            # Pool decided validation pairs so abstaining folds don't skew precision.
            "mean_precision": (
                total_correct_decided_pair_count / total_decided_pair_count
                if total_decided_pair_count
                else None
            ),
            "fold_precisions": fold_precisions,
            "mean_prediction_coverage": (
                total_decided_pair_count / total_validation_pair_count
            ),
            "feature_metrics": cross_validation_feature_metrics,
            "folds_detail": fold_metadata,
        },
        "configuration": schema,
        "report_num_queries": report_num_queries,
        "report_seed": report_seed,
        "grade_column": grade_column,
    }
    (run_path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return PairwiseTrainedArtifact(path=run_path, metadata=metadata)


@dataclass
class PairwiseTrainedArtifact:
    path: Path
    metadata: dict[str, Any]
