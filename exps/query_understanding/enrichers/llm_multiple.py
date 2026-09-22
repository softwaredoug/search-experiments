from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from cheat_at_search.enrich.enrich import AutoEnricher
from pydantic import Field, create_model

from exps.query_understanding.enrichers.category_values import (
    append_aliases,
    schema_values,
)

SYSTEM_PROMPT = (
    "You are a helpful furniture shopping agent that helps users construct search queries."
)


def _model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


class LLMMultipleEnricher:
    def __init__(
        self,
        *,
        field: str,
        vocabulary: list[str],
        prompt: str,
        model: str = "gpt-5-mini",
        reasoning: str | None = None,
        temperature: float | None = None,
        verbosity: str | None = None,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("llm_multiple enrichment requires a non-empty prompt.")
        self.field = field
        self.vocabulary = list(vocabulary)
        self.prompt_template = prompt
        self.model = _model_name(model)
        self.reasoning = reasoning
        self.temperature = temperature
        self.verbosity = verbosity
        allowed_values, self.schema_aliases = schema_values(self.vocabulary)
        allowed_values = tuple(dict.fromkeys([*allowed_values, "Unknown"]))
        category_type = Literal[allowed_values]
        self.response_model = create_model(
            "CategoryEnrichmentMultiple",
            categories=(
                list[category_type],
                Field(
                    ...,
                    description=(
                        f"The {self.field} values to use for filtering or boosting results."
                    ),
                ),
            ),
        )
        self.enricher = AutoEnricher(
            model=self.model,
            system_prompt=SYSTEM_PROMPT,
            response_model=self.response_model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning,
            verbosity=self.verbosity,
        )
        self._cache: dict[str, list[str]] = {}

    def enrich(self, query: str) -> list[str]:
        if query in self._cache:
            return list(self._cache[query])
        prompt = self.prompt_template.format(field=self.field, query=query)
        prompt = append_aliases(prompt, self.field, self.schema_aliases)
        response = self.enricher.enrich(prompt)
        values = getattr(response, "categories", []) if response is not None else []
        if not isinstance(values, list):
            values = list(values) if values else []
        categories = []
        for value in values:
            value = str(value)
            value = self.schema_aliases.get(value, value)
            if value == "Unknown" or value in categories:
                continue
            categories.append(value)
        self._cache[query] = categories
        return list(categories)

    @property
    def cache_key(self) -> str:
        payload: dict[str, Any] = {
            "type": "llm_multiple",
            "field": self.field,
            "vocabulary": self.vocabulary,
            "model": self.model,
            "reasoning": self.reasoning,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": self.prompt_template,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()


def make_llm_multiple_enricher(
    *,
    field: str,
    vocabulary: list[str],
    prompt: str,
    model: str = "gpt-5-mini",
    reasoning: str | None = None,
    params: dict[str, Any] | None = None,
) -> LLMMultipleEnricher:
    params = params or {}
    return LLMMultipleEnricher(
        field=field,
        vocabulary=vocabulary,
        prompt=prompt,
        model=model,
        reasoning=reasoning,
        temperature=params.get("temperature"),
        verbosity=params.get("verbosity"),
    )
