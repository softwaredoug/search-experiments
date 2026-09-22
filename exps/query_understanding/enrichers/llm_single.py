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


class LLMSingleEnricher:
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
        self.field = field
        self.vocabulary = list(vocabulary)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("llm_single enrichment requires a non-empty prompt.")
        self.prompt_template = prompt
        self.model = _model_name(model)
        self.reasoning = reasoning
        self.temperature = temperature
        self.verbosity = verbosity
        allowed_values, self.schema_aliases = schema_values(self.vocabulary)
        allowed_values = tuple(dict.fromkeys([*allowed_values, "Unknown"]))
        category_type = Literal[allowed_values]
        self.response_model = create_model(
            "CategoryEnrichment",
            **{
                "category": (
                    category_type,
                    Field(
                        ...,
                        description=(
                            f"The {self.field} value to use for filtering or boosting results."
                        ),
                    ),
                )
            },
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
        value = getattr(response, "category", None) if response is not None else None
        value = self.schema_aliases.get(value, value)
        categories = [] if value in (None, "Unknown") else [str(value)]
        self._cache[query] = categories
        return list(categories)

    @property
    def cache_key(self) -> str:
        payload: dict[str, Any] = {
            "type": "llm_single",
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


def make_llm_single_enricher(
    *,
    field: str,
    vocabulary: list[str],
    prompt: str,
    model: str = "gpt-5-mini",
    reasoning: str | None = None,
    params: dict[str, Any] | None = None,
) -> LLMSingleEnricher:
    params = params or {}
    return LLMSingleEnricher(
        field=field,
        vocabulary=vocabulary,
        prompt=prompt,
        model=model,
        reasoning=reasoning,
        temperature=params.get("temperature"),
        verbosity=params.get("verbosity"),
    )
