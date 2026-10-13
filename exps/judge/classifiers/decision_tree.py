from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sklearn.tree import DecisionTreeClassifier


def normalize_params(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(params)
    if unknown:
        raise ValueError(
            "Unsupported decision_tree model.params option(s): "
            + ", ".join(sorted(unknown))
        )
    return {}


def build(_params: Mapping[str, Any]) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(random_state=42)
