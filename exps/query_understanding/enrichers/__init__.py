from __future__ import annotations

from typing import Any

from exps.query_understanding.enrichers.dummy import (
    DummyEnricher,
    make_dummy_enricher,
)
from exps.query_understanding.enrichers.choice_single import make_choice_single_enricher
from exps.query_understanding.enrichers.choice_single_jev import JevChoiceSingleEnricher
from exps.query_understanding.enrichers.choice_single_openai import (
    OpenAIChoiceSingleEnricher,
)
from exps.query_understanding.enrichers.llm_single import (
    LLMSingleEnricher,
    make_llm_single_enricher,
)
from exps.query_understanding.enrichers.llm_multiple import (
    LLMMultipleEnricher,
    make_llm_multiple_enricher,
)
from exps.query_understanding.enrichers.protocol import Enricher


def make_enricher(
    config: dict[str, Any] | None,
    *,
    field: str,
    vocabulary: list[str],
    model: str = "gpt-5-mini",
    reasoning: str | None = None,
) -> Enricher:
    config = config or {}
    enrichment_type = config.get("type")
    params = config.get("params") or {}
    if enrichment_type == "dummy":
        return make_dummy_enricher(vocabulary)
    if enrichment_type == "choice_single":
        prompt = params.get("prompt")
        choices = params.get("choices")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "choice_single enrichment requires params.prompt as a non-empty template."
            )
        if choices is None:
            raise ValueError("choice_single enrichment requires params.choices.")
        return make_choice_single_enricher(
            field=field,
            vocabulary=vocabulary,
            choices=choices,
            prompt=prompt,
            model=model,
            reasoning=reasoning,
            params=params,
        )
    if enrichment_type == "llm_single":
        prompt = params.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "llm_single enrichment requires params.prompt as a non-empty template."
            )
        return make_llm_single_enricher(
            field=field,
            vocabulary=vocabulary,
            prompt=prompt,
            model=model,
            reasoning=reasoning,
            params=params,
        )
    if enrichment_type == "llm_multiple":
        prompt = params.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "llm_multiple enrichment requires params.prompt as a non-empty template."
            )
        return make_llm_multiple_enricher(
            field=field,
            vocabulary=vocabulary,
            prompt=prompt,
            model=model,
            reasoning=reasoning,
            params=params,
        )
    raise ValueError(
        "Supported enrichment engines are dummy, choice_single, llm_single, and llm_multiple; "
        f"received {enrichment_type!r}."
    )


__all__ = [
    "DummyEnricher",
    "JevChoiceSingleEnricher",
    "OpenAIChoiceSingleEnricher",
    "Enricher",
    "LLMSingleEnricher",
    "LLMMultipleEnricher",
    "make_dummy_enricher",
    "make_choice_single_enricher",
    "make_enricher",
    "make_llm_single_enricher",
    "make_llm_multiple_enricher",
]
