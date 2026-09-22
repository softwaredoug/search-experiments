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
from exps.query_understanding.enrichers.choice_single import _choice_vocabulary

SYSTEM_PROMPT = (
    "You are a helpful furniture shopping agent that helps users construct search queries."
)


def _model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


class OpenAIChoiceSingleEnricher:
    def __init__(
        self,
        *,
        field: str,
        vocabulary: list[str],
        choices: dict[str, str],
        prompt: str,
        model: str,
        reasoning: str | None,
        pad_missing_choices: bool,
        temperature: float | None = None,
        verbosity: str | None = None,
    ):
        self.field = field
        self.vocabulary = list(vocabulary)
        self.prompt_template = prompt
        self.model = _model_name(model)
        self.reasoning = reasoning
        self.temperature = temperature
        self.verbosity = verbosity
        self.pad_missing_choices = pad_missing_choices
        self.choices = dict(choices)

        allowed_vocabulary = _choice_vocabulary(
            self.vocabulary, self.choices, self.pad_missing_choices
        )
        allowed_values, self.schema_aliases = schema_values(allowed_vocabulary)
        if not allowed_values:
            raise ValueError("choice_single choices must contain a vocabulary value.")
        allowed_values.append("Unknown")
        choice_type = Literal[tuple(dict.fromkeys(allowed_values))]
        self.response_model = create_model(
            "ChoiceEnrichment",
            choice=(
                choice_type,
                Field(..., description=f"The {self.field} choice for the query."),
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

    def _prompt(self, query: str) -> str:
        prompt = self.prompt_template.format(field=self.field, query=query)
        prompt_choices = dict(self.choices)
        prompt_choices.setdefault("Unknown", "No classification applies.")
        choice_lines = "\n".join(
            f"- {choice}: {description}"
            for choice, description in prompt_choices.items()
        )
        prompt = f"{prompt}\n\nChoices:\n{choice_lines}"
        return append_aliases(prompt, self.field, self.schema_aliases)

    def enrich(self, query: str) -> list[str]:
        if query in self._cache:
            return list(self._cache[query])
        response = self.enricher.enrich(self._prompt(query))
        value = getattr(response, "choice", None) if response is not None else None
        value = self.schema_aliases.get(value, value)
        categories = [] if value in (None, "Unknown") else [str(value)]
        self._cache[query] = categories
        return list(categories)

    @property
    def cache_key(self) -> str:
        payload: dict[str, Any] = {
            "type": "choice_single_openai",
            "field": self.field,
            "vocabulary": self.vocabulary,
            "choices": self.choices,
            "pad_missing_choices": self.pad_missing_choices,
            "model": self.model,
            "reasoning": self.reasoning,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt_template": self.prompt_template,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
