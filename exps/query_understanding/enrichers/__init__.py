from __future__ import annotations

from typing import Any

from exps.query_understanding.enrichers.dummy import (
    DummyEnricher,
    make_dummy_enricher,
)
from exps.query_understanding.enrichers.cached_choice_single_jev import (
    CachedJevChoiceSingleEnricher,
)
from exps.query_understanding.enrichers.cached_jev_choice_multiple import (
    CachedJevChoiceMultipleEnricher,
)
from exps.query_understanding.enrichers.choice_single import (
    make_jev_choice_single_enricher,
    make_llm_choice_enricher,
)
from exps.query_understanding.enrichers.choice_single_jev import JevChoiceSingleEnricher
from exps.query_understanding.enrichers.choice_single_openai import (
    OpenAIChoiceSingleEnricher,
)
from exps.query_understanding.enrichers.jev_choice_multiple import (
    JevChoiceMultipleEnricher,
    make_jev_choice_multiple_enricher,
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
from exps.query_understanding.enrichers.jev_bm25_then_select import (
    JevBM25ThenSelectEnricher,
    JevBM25ThenSelectMultipleEnricher,
    make_jev_bm25_then_select_enricher,
    make_jev_bm25_then_select_multiple_enricher,
)


def make_enricher(
    config: dict[str, Any] | None,
    *,
    field: str,
    vocabulary: list[str],
    model: str = "gpt-5-mini",
    reasoning: str | None = None,
    corpus=None,
    no_cache: bool = False,
) -> Enricher:
    config = config or {}
    enrichment_type = config.get("type")
    params = config.get("params") or {}
    if enrichment_type == "dummy":
        return make_dummy_enricher(vocabulary)
    if enrichment_type in {"llm_choice", "jev_choice_single"}:
        prompt = params.get("prompt")
        choices = params.get("choices")
        factory = (
            make_llm_choice_enricher
            if enrichment_type == "llm_choice"
            else make_jev_choice_single_enricher
        )
        kwargs = {
            "field": field,
            "vocabulary": vocabulary,
            "choices": choices,
            "prompt": prompt,
            "model": model,
            "reasoning": reasoning,
            "params": params,
        }
        if enrichment_type == "jev_choice_single":
            kwargs["no_cache"] = no_cache
        return factory(**kwargs)
    if enrichment_type == "jev_choice_multiple":
        return make_jev_choice_multiple_enricher(
            field=field,
            vocabulary=vocabulary,
            choices=params.get("choices"),
            prompt=params.get("prompt"),
            model=model,
            reasoning=reasoning,
            params=params,
            no_cache=no_cache,
        )
    if enrichment_type == "jev_bm25_then_select":
        if corpus is None:
            raise ValueError(
                "jev_bm25_then_select requires the corpus for its candidate BM25 search."
            )
        return make_jev_bm25_then_select_enricher(
            corpus=corpus,
            field=field,
            params=params,
            model=model,
            no_cache=no_cache,
        )
    if enrichment_type == "jev_bm25_then_select_multiple":
        if corpus is None:
            raise ValueError(
                "jev_bm25_then_select_multiple requires the corpus for its candidate BM25 search."
            )
        return make_jev_bm25_then_select_multiple_enricher(
            corpus=corpus,
            field=field,
            params=params,
            model=model,
            no_cache=no_cache,
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
        "Supported enrichment engines are dummy, llm_choice, jev_choice_single, "
        "jev_choice_multiple, jev_bm25_then_select, "
        "jev_bm25_then_select_multiple, llm_single, and llm_multiple; "
        f"received {enrichment_type!r}."
    )


__all__ = [
    "DummyEnricher",
    "CachedJevChoiceSingleEnricher",
    "CachedJevChoiceMultipleEnricher",
    "JevChoiceMultipleEnricher",
    "JevChoiceSingleEnricher",
    "JevBM25ThenSelectEnricher",
    "JevBM25ThenSelectMultipleEnricher",
    "OpenAIChoiceSingleEnricher",
    "Enricher",
    "LLMSingleEnricher",
    "LLMMultipleEnricher",
    "make_dummy_enricher",
    "make_llm_choice_enricher",
    "make_jev_choice_single_enricher",
    "make_jev_choice_multiple_enricher",
    "make_jev_bm25_then_select_enricher",
    "make_jev_bm25_then_select_multiple_enricher",
    "make_enricher",
    "make_llm_single_enricher",
    "make_llm_multiple_enricher",
]
