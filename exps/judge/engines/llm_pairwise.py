from __future__ import annotations

from typing import Literal

from cheat_at_search.enrich.enrich import AutoEnricher
from pydantic import Field, create_model

from exps.judge.engines.common import PairwiseJudgeResult
from exps.judge.engines.llm_pointwise import _openai_model_name


class LLMPairwiseJudgeEngine:
    """Use AutoEnricher to select which of two documents is more relevant."""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        explicit_neither: bool = False,
        no_cache: bool = False,
    ):
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("strategy.params.system_prompt must be a non-empty string.")
        if not isinstance(explicit_neither, bool):
            raise ValueError("strategy.params.explicit_neither must be a boolean.")
        self.model = _openai_model_name(model)
        self.explicit_neither = explicit_neither
        self.no_cache = no_cache
        choices = ("lhs", "rhs", "neither") if explicit_neither else ("lhs", "rhs")
        winner_literal = Literal.__getitem__(choices)
        self.response_model = create_model(
            "LLMPairwiseJudgeOutcome",
            winner=(
                winner_literal,
                Field(description="Which document is more relevant: lhs, rhs, or neither."),
            ),
        )
        self.enricher = AutoEnricher(
            model=self.model,
            system_prompt=system_prompt,
            response_model=self.response_model,
        )

    def judge(
        self,
        *,
        prompt: str,
        question: str,
        prompt_has_question: bool = False,
        **_kwargs,
    ) -> PairwiseJudgeResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Judge prompt must be a non-empty string.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Judge question must be a non-empty string.")
        parts = [prompt]
        if not prompt_has_question:
            parts.append(question)
        request = "\n\n".join(parts)
        if self.no_cache:
            response = self.enricher.enricher.enrich(request)
        else:
            response = self.enricher.enrich(request)
        outcome = getattr(response, "winner", None) if response is not None else None
        allowed = {"lhs", "rhs"}
        if self.explicit_neither:
            allowed.add("neither")
        if outcome not in allowed:
            raise ValueError(f"LLM returned an invalid pairwise outcome: {outcome!r}.")
        return PairwiseJudgeResult(outcome=outcome)

    @property
    def cache_backend(self):
        return self.enricher.cached_enricher

    def flush_cache(self) -> None:
        if not self.no_cache:
            self.cache_backend.save_cache()
