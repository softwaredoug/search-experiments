from __future__ import annotations

import hashlib
import json
from typing import Any

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Choice, TypeSafeClient

from exps.query_understanding.enrichers.choice_single import _choice_vocabulary


def _model_name(model: str) -> str:
    if model == "jev":
        return "jev-latest"
    if model.startswith("jev/"):
        return model.replace("/", "-", 1)
    return model


class JevChoiceSingleEnricher:
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
    ):
        self.field = field
        self.vocabulary = list(vocabulary)
        self.prompt_template = prompt
        self.model = _model_name(model)
        self.reasoning = reasoning
        self.pad_missing_choices = pad_missing_choices
        self.choices = dict(choices)
        allowed_vocabulary = _choice_vocabulary(
            self.vocabulary, self.choices, self.pad_missing_choices
        )
        self.criteria = {
            value: self.choices.get(value)
            for value in allowed_vocabulary
        }
        self.criteria["Unknown"] = self.choices.get(
            "Unknown", "No classification applies."
        )
        if len(self.criteria) > 255:
            raise ValueError("Jev choice_single supports at most 255 choices.")
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
        value = getattr(answer, "choice", None)
        categories = [] if value in (None, "Unknown") else [str(value)]
        self._cache[query] = categories
        return list(categories)

    @property
    def cache_key(self) -> str:
        payload: dict[str, Any] = {
            "type": "choice_single_jev",
            "field": self.field,
            "vocabulary": self.vocabulary,
            "criteria": self.criteria,
            "pad_missing_choices": self.pad_missing_choices,
            "model": self.model,
            "user_prompt_template": self.prompt_template,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
