from __future__ import annotations

import math
from typing import Any

import yaml


MAX_JEV_CHOICE_COUNT = 255


def _parse_choices(choices: Any) -> dict[str, str | None]:
    if isinstance(choices, str):
        choices = yaml.safe_load(choices)
    if choices is None:
        return {}
    if not isinstance(choices, dict):
        raise ValueError("choice enrichment requires params.choices as a mapping.")
    parsed = {}
    for choice, description in choices.items():
        if not isinstance(choice, str) or not choice.strip():
            raise ValueError("choice options must have non-empty string keys.")
        if description is not None and (
            not isinstance(description, str) or not description.strip()
        ):
            raise ValueError(f"choice option {choice!r} must have a description.")
        parsed[choice] = description.strip() if isinstance(description, str) else None
    return parsed


def _prepare_choices(
    choices: Any, vocabulary: list[str], pad_missing_choices: bool
) -> dict[str, str | None]:
    parsed_choices = _parse_choices(choices)
    if pad_missing_choices:
        return parsed_choices
    return {
        value: description
        for value, description in parsed_choices.items()
        if value in vocabulary or value == "Unknown"
    }


def _choice_vocabulary(
    vocabulary: list[str], choices: dict[str, str | None], pad_missing_choices: bool
) -> list[str]:
    if pad_missing_choices:
        return list(vocabulary)
    return [value for value in vocabulary if value in choices]


def _prepare_choice_options(
    *,
    choices: Any,
    vocabulary: list[str],
    pad_missing_choices: bool,
    max_values: int | None = None,
) -> tuple[list[str], dict[str, str | None]]:
    parsed_choices = _parse_choices(choices)
    choice_vocabulary = list(vocabulary)
    if not parsed_choices:
        choice_vocabulary = [value for value in choice_vocabulary if value != "Unknown"]
        if max_values is not None:
            choice_vocabulary = choice_vocabulary[:max_values]
        parsed_choices = {value: None for value in choice_vocabulary}
    prepared_choices = _prepare_choices(
        parsed_choices, choice_vocabulary, pad_missing_choices
    )
    return choice_vocabulary, prepared_choices


def _choice_confidence_threshold(params: dict[str, Any]) -> float | None:
    confidence_threshold = params.get("confidence_threshold")
    if confidence_threshold is None:
        return None
    try:
        confidence_threshold = float(confidence_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence_threshold must be a number between 0 and 1.") from exc
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be a number between 0 and 1.")
    return confidence_threshold


def _validate_choice_prompt(prompt: str, engine_name: str) -> None:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{engine_name} enrichment requires a non-empty prompt.")


def make_llm_choice_enricher(
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
    _validate_choice_prompt(prompt, "llm_choice")
    pad_missing_choices = bool(params.get("pad_missing_choices", False))
    configured_model = str(params.get("model", model))
    provider = configured_model.split("/", 1)[0].lower()
    if "/" in configured_model and provider != "openai":
        raise ValueError(
            "llm_choice supports OpenAI models only; "
            f"received {configured_model!r}."
        )
    if params.get("confidence_threshold") is not None:
        raise ValueError("confidence_threshold is only supported for Jev models.")
    choice_vocabulary, prepared_choices = _prepare_choice_options(
        choices=choices,
        vocabulary=vocabulary,
        pad_missing_choices=pad_missing_choices,
    )
    from exps.query_understanding.enrichers.choice_single_openai import (
        OpenAIChoiceSingleEnricher,
    )

    return OpenAIChoiceSingleEnricher(
        field=field,
        vocabulary=choice_vocabulary,
        choices=prepared_choices,
        prompt=prompt,
        model=configured_model,
        reasoning=reasoning,
        pad_missing_choices=pad_missing_choices,
        temperature=params.get("temperature"),
        verbosity=params.get("verbosity"),
    )


def make_jev_choice_single_enricher(
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
    _validate_choice_prompt(prompt, "jev_choice_single")
    configured_model = str(params.get("model", model))
    provider = configured_model.split("/", 1)[0].lower()
    if provider != "jev":
        raise ValueError(
            "jev_choice_single requires a Jev model (jev/*); "
            f"received {configured_model!r}."
        )
    pad_missing_choices = bool(params.get("pad_missing_choices", False))
    confidence_threshold = _choice_confidence_threshold(params)
    choice_vocabulary, prepared_choices = _prepare_choice_options(
        choices=choices,
        vocabulary=vocabulary,
        pad_missing_choices=pad_missing_choices,
        max_values=MAX_JEV_CHOICE_COUNT - 1,
    )
    from exps.query_understanding.enrichers.cached_choice_single_jev import (
        CachedJevChoiceSingleEnricher,
    )

    return CachedJevChoiceSingleEnricher(
        field=field,
        vocabulary=choice_vocabulary,
        choices=prepared_choices,
        prompt=prompt,
        model=configured_model,
        reasoning=reasoning,
        pad_missing_choices=pad_missing_choices,
        confidence_threshold=confidence_threshold,
    )


__all__ = ["make_llm_choice_enricher", "make_jev_choice_single_enricher"]
