from __future__ import annotations

import math
from typing import Any

import yaml


def _parse_choices(choices: Any) -> dict[str, str]:
    if isinstance(choices, str):
        choices = yaml.safe_load(choices)
    if not isinstance(choices, dict):
        raise ValueError("choice_single enrichment requires params.choices as a mapping.")
    parsed = {}
    for choice, description in choices.items():
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError("choice_single choices must have non-empty string keys.")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"choice_single choice {choice!r} must have a description.")
        parsed[choice] = description.strip()
    if not parsed:
        raise ValueError("choice_single enrichment requires at least one choice.")
    return parsed


def _prepare_choices(
    choices: Any, vocabulary: list[str], pad_missing_choices: bool
) -> dict[str, str]:
    parsed_choices = _parse_choices(choices)
    if pad_missing_choices:
        return parsed_choices
    return {
        value: description
        for value, description in parsed_choices.items()
        if value in vocabulary or value == "Unknown"
    }


def _choice_vocabulary(
    vocabulary: list[str], choices: dict[str, str], pad_missing_choices: bool
) -> list[str]:
    if pad_missing_choices:
        return list(vocabulary)
    return [value for value in vocabulary if value in choices]


def make_choice_single_enricher(
    *,
    field: str,
    vocabulary: list[str],
    choices: Any,
    prompt: str,
    model: str = "gpt-5-mini",
    reasoning: str | None = None,
    params: dict[str, Any] | None = None,
):
    params = params or {}
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("choice_single enrichment requires a non-empty prompt.")
    pad_missing_choices = bool(params.get("pad_missing_choices", False))
    confidence_threshold = params.get("confidence_threshold")
    if confidence_threshold is not None:
        try:
            confidence_threshold = float(confidence_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("confidence_threshold must be a number between 0 and 1.") from exc
        if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be a number between 0 and 1.")
    prepared_choices = _prepare_choices(choices, vocabulary, pad_missing_choices)
    common = {
        "field": field,
        "vocabulary": vocabulary,
        "choices": prepared_choices,
        "prompt": prompt,
        "model": params.get("model", model),
        "reasoning": reasoning,
        "pad_missing_choices": pad_missing_choices,
        "confidence_threshold": confidence_threshold,
    }
    configured_model = str(common["model"])
    provider = configured_model.split("/", 1)[0].lower()
    if "/" in configured_model and provider == "jev":
        from exps.query_understanding.enrichers.choice_single_jev import (
            JevChoiceSingleEnricher,
        )

        return JevChoiceSingleEnricher(**common)
    if "/" not in configured_model or provider == "openai":
        if confidence_threshold is not None:
            raise ValueError(
                "confidence_threshold is only supported for Jev choice_single models."
            )
        from exps.query_understanding.enrichers.choice_single_openai import (
            OpenAIChoiceSingleEnricher,
        )

        openai_common = {
            key: value for key, value in common.items() if key != "confidence_threshold"
        }
        return OpenAIChoiceSingleEnricher(
            **openai_common,
            temperature=params.get("temperature"),
            verbosity=params.get("verbosity"),
        )
    raise ValueError(
        "choice_single supports unprefixed/OpenAI models and Jev (jev/*) models; "
        f"received {common['model']!r}."
    )


__all__ = ["make_choice_single_enricher"]
