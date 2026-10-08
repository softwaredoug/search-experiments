from __future__ import annotations

import hashlib
import json
from typing import Any

from cheat_at_search.enrich.enrich import AutoEnricher
from pydantic import Field, create_model


def _model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


class DecisionGenerator:
    """Generate query-specific yes/no decision questions with an LLM."""

    def __init__(
        self,
        *,
        system_prompt: str,
        prompt: str,
        model: str,
        reasoning: str | None = None,
        temperature: float | None = None,
        verbosity: str | None = None,
        no_cache: bool = False,
    ):
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError(
                "decision_engine.system_prompt must be a non-empty string."
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("decision_engine.prompt must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("decision_engine.model must be a non-empty string.")

        self.system_prompt = system_prompt
        self.prompt_template = prompt
        self.model = _model_name(model.strip())
        self.reasoning = reasoning
        self.temperature = temperature
        self.verbosity = verbosity
        self.no_cache = no_cache
        self.response_model = create_model(
            "BagOfDecisionsQuestions",
            decisions=(
                list[str],
                Field(
                    ...,
                    description=(
                        "A list of concise yes/no questions whose affirmative "
                        "answer indicates relevance to the search query."
                    ),
                ),
            ),
        )
        self.enricher = AutoEnricher(
            model=self.model,
            system_prompt=self.system_prompt,
            response_model=self.response_model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning,
            verbosity=self.verbosity,
        )

    def generate(self, query: str) -> list[str]:
        try:
            prompt = self.prompt_template.format(query=query)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("decision_engine.prompt may use {query}.") from exc

        if self.no_cache:
            # AutoEnricher.enrich goes through its persistent cached client.
            response = self.enricher.enricher.enrich(prompt)
        else:
            response = self.enricher.enrich(prompt)
        decisions = getattr(response, "decisions", None) if response is not None else None
        if not isinstance(decisions, (list, tuple)):
            decisions = []
        normalized: list[str] = []
        for decision in decisions:
            if (
                isinstance(decision, str)
                and decision.strip()
                and decision.strip() not in normalized
            ):
                normalized.append(decision.strip())
        return list(normalized)

    @property
    def cache_key(self) -> str:
        payload: dict[str, Any] = {
            "type": "decision_generator",
            "model": self.model,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt_template,
            "reasoning": self.reasoning,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            "no_cache": self.no_cache,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
