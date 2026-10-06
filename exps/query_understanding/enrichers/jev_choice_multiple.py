from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Choice, TypeSafeClient

from exps.query_understanding.enrichers.choice_single import (
    MAX_JEV_CHOICE_COUNT,
    _choice_vocabulary,
    _parse_choices,
    _prepare_choice_options,
    _validate_choice_prompt,
)


def _model_name(model: str) -> str:
    if model == "jev":
        return "jev-latest"
    if model.startswith("jev/"):
        return model.split("/", 1)[1]
    return model


def _parse_threshold(value: Any) -> float:
    if value is None:
        raise ValueError("jev_choice_multiple requires params.threshold.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("threshold must be a number between 0 and 1.") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be a number between 0 and 1.")
    return threshold


class JevChoiceMultipleEnricher:
    def __init__(
        self,
        *,
        field: str,
        vocabulary: list[str],
        choices: dict[str, str | None],
        prompt: str,
        model: str,
        threshold: float,
        reasoning: str | None,
        pad_missing_choices: bool,
    ):
        self.field = field
        self.vocabulary = list(vocabulary)
        self.choices = dict(choices)
        self.prompt_template = prompt
        self.model = _model_name(model)
        self.threshold = threshold
        self.reasoning = reasoning
        self.pad_missing_choices = pad_missing_choices

        allowed_vocabulary = _choice_vocabulary(
            self.vocabulary, self.choices, self.pad_missing_choices
        )
        self.criteria = {value: self.choices.get(value) for value in allowed_vocabulary}
        if "Unknown" in self.choices and "Unknown" not in self.criteria:
            self.criteria["Unknown"] = self.choices["Unknown"]
        if not self.criteria:
            raise ValueError("jev_choice_multiple requires at least one choice option.")
        if len(self.criteria) > MAX_JEV_CHOICE_COUNT:
            raise ValueError(
                f"jev_choice_multiple supports at most {MAX_JEV_CHOICE_COUNT} choices."
            )

        self.client = TypeSafeClient(
            api_key=key_for_provider("typesafe"),
            model=self.model,
        )
        self._cache: dict[str, list[str]] = {}

    def _prompt(self, query: str) -> str:
        return self.prompt_template.format(field=self.field, query=query)

    def enrich(self, query: str) -> list[str]:
        if query in self._cache:
            return list(self._cache[query])
        response = self.client.system_one(
            state=query,
            questions={
                self.field: Choice(
                    instructions=self._prompt(query),
                    criteria=self.criteria,
                )
            },
        )
        answer = response.choices[self.field]
        probabilities = getattr(answer, "probabilities", None)
        categories = sorted(
            {
                str(category)
                for category, probability in (probabilities or {}).items()
                if category in self.criteria
                and category != "Unknown"
                and isinstance(probability, (int, float))
                and math.isfinite(probability)
                and probability > self.threshold
            }
        )
        self._cache[query] = categories
        return list(categories)

    @property
    def cache_key(self) -> str:
        payload: dict[str, Any] = {
            "type": "jev_choice_multiple",
            "field": self.field,
            "vocabulary": self.vocabulary,
            "criteria": self.criteria,
            "pad_missing_choices": self.pad_missing_choices,
            "threshold": self.threshold,
            "model": self.model,
            "user_prompt_template": self.prompt_template,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()


def make_jev_choice_multiple_enricher(
    *,
    field: str,
    vocabulary: list[str],
    choices: Any,
    prompt: str,
    model: str = "gpt-5-mini",
    reasoning: str | None = None,
    params: dict[str, Any] | None = None,
    no_cache: bool = False,
):
    params = params or {}
    _validate_choice_prompt(prompt, "jev_choice_multiple")
    configured_model = str(params.get("model", model))
    if configured_model.split("/", 1)[0].lower() != "jev":
        raise ValueError(
            "jev_choice_multiple requires a Jev model (jev/*); "
            f"received {configured_model!r}."
        )
    threshold = _parse_threshold(params.get("threshold"))
    pad_missing_choices = bool(params.get("pad_missing_choices", False))
    parsed_choices = _parse_choices(choices)
    max_category_values = MAX_JEV_CHOICE_COUNT
    if pad_missing_choices and "Unknown" in parsed_choices:
        max_category_values -= 1
    choice_vocabulary, prepared_choices = _prepare_choice_options(
        choices=parsed_choices,
        vocabulary=vocabulary,
        pad_missing_choices=pad_missing_choices,
        max_values=max_category_values,
        reserve_unknown=False,
    )

    from exps.query_understanding.enrichers.cached_jev_choice_multiple import (
        CachedJevChoiceMultipleEnricher,
    )

    return CachedJevChoiceMultipleEnricher(
        field=field,
        vocabulary=choice_vocabulary,
        choices=prepared_choices,
        prompt=prompt,
        model=configured_model,
        threshold=threshold,
        reasoning=reasoning,
        pad_missing_choices=pad_missing_choices,
        no_cache=no_cache,
    )


__all__ = ["JevChoiceMultipleEnricher", "make_jev_choice_multiple_enricher"]
