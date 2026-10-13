from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol

from sklearn.base import ClassifierMixin

from exps.judge.classifiers import decision_tree, gradient_boosted_tree


class ClassifierModule(Protocol):
    """Contract implemented by each pairwise sklearn classifier module."""

    def normalize_params(self, params: Mapping[str, Any]) -> dict[str, Any]: ...

    def build(self, params: Mapping[str, Any]) -> ClassifierMixin: ...


_CLASSIFIERS: dict[str, ClassifierModule] = {
    "decision_tree": decision_tree,
    "gradient_boosted_tree": gradient_boosted_tree,
}


def normalize_model_config(raw_model: Any) -> dict[str, Any]:
    """Validate shared model settings and delegate estimator params by type."""
    if raw_model is None:
        raw_model = {"type": "decision_tree"}
    if not isinstance(raw_model, dict):
        raise ValueError("strategy.params.model must be a mapping with a 'type'.")
    unknown = set(raw_model) - {"type", "params"}
    if unknown:
        raise ValueError(
            "Unsupported strategy.params.model option(s): " + ", ".join(sorted(unknown))
        )

    model_type = raw_model.get("type", "decision_tree")
    if not isinstance(model_type, str) or model_type not in _CLASSIFIERS:
        supported = ", ".join(repr(name) for name in _CLASSIFIERS)
        raise ValueError(f"strategy.params.model.type must be one of {supported}.")

    raw_params = raw_model.get("params", {})
    if raw_params is None:
        raw_params = {}
    if not isinstance(raw_params, dict):
        raise ValueError("strategy.params.model.params must be a mapping.")

    prediction_threshold = raw_params.get("prediction_threshold")
    if prediction_threshold is not None and (
        isinstance(prediction_threshold, bool)
        or not isinstance(prediction_threshold, (int, float))
        or not math.isfinite(prediction_threshold)
        or not 0 <= prediction_threshold <= 1
    ):
        raise ValueError(
            "strategy.params.model.params.prediction_threshold must be between zero and one."
        )

    estimator_params = {
        name: value
        for name, value in raw_params.items()
        if name != "prediction_threshold"
    }
    normalized_params = _CLASSIFIERS[model_type].normalize_params(estimator_params)
    if prediction_threshold is not None:
        normalized_params["prediction_threshold"] = float(prediction_threshold)

    return {"type": model_type, "params": normalized_params}


def build_classifier(model_config: Mapping[str, Any]) -> ClassifierMixin:
    """Build the sklearn estimator described by a normalized model config."""
    try:
        classifier = _CLASSIFIERS[model_config["type"]]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported pairwise classifier type: {model_config.get('type')!r}."
        ) from exc
    estimator_params = {
        name: value
        for name, value in model_config.get("params", {}).items()
        if name != "prediction_threshold"
    }
    return classifier.build(estimator_params)


__all__ = ["build_classifier", "normalize_model_config"]
