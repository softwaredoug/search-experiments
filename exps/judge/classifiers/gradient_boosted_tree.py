from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.ensemble import GradientBoostingClassifier


def normalize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(params) - {"n_estimators", "max_depth"}
    if unknown:
        raise ValueError(
            "Unsupported gradient_boosted_tree model.params option(s): "
            + ", ".join(sorted(unknown))
        )

    n_estimators = params.get("n_estimators", 100)
    if (
        isinstance(n_estimators, bool)
        or not isinstance(n_estimators, int)
        or n_estimators < 1
    ):
        raise ValueError(
            "gradient_boosted_tree model.params.n_estimators must be a positive integer."
        )

    max_depth = params.get("max_depth", 3)
    if max_depth is not None and (
        isinstance(max_depth, bool)
        or not isinstance(max_depth, int)
        or max_depth < 1
    ):
        raise ValueError(
            "gradient_boosted_tree model.params.max_depth must be a positive integer or null."
        )

    return {"n_estimators": n_estimators, "max_depth": max_depth}


def build(params: Mapping[str, Any]) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=42,
    )
