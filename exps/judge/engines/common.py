from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class JudgeResult:
    grade: float
    probability: float | None = None


def grade_value(value) -> float:
    if isinstance(value, bool):
        raise ValueError("Judgment grades must be finite numeric values.")
    try:
        grade = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Judgment grades must be finite numeric values.") from exc
    if not math.isfinite(grade):
        raise ValueError("Judgment grades must be finite numeric values.")
    return grade


def normalize_rubric(rubric: Mapping) -> dict[float, str]:
    if not isinstance(rubric, Mapping) or not rubric:
        raise ValueError("strategy.params.rubric must be a non-empty grade mapping.")
    normalized: dict[float, str] = {}
    for raw_grade, description in rubric.items():
        grade = grade_value(raw_grade)
        if not isinstance(description, str) or not description.strip():
            raise ValueError("Every rubric grade must have a non-empty description.")
        if grade in normalized:
            raise ValueError(f"Duplicate rubric grade: {raw_grade!r}")
        normalized[grade] = description.strip()
    return normalized


def rubric_text(rubric: Mapping[float, str]) -> str:
    return "\n".join(
        f"{grade:g}: {description}"
        for grade, description in sorted(rubric.items(), reverse=True)
    )


def rubric_criteria(rubric: Mapping[float, str]) -> dict[str, str]:
    return {f"{grade:g}": description for grade, description in rubric.items()}


def choice_grade(choice, rubric: Mapping[float, str]) -> float:
    try:
        grade = grade_value(choice)
    except ValueError as exc:
        raise ValueError(f"Jev returned a choice outside the rubric: {choice!r}.") from exc
    if grade not in rubric:
        raise ValueError(f"Jev returned a choice outside the rubric: {choice!r}.")
    return grade
