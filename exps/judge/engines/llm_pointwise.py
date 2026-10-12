from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from cheat_at_search.enrich.enrich import AutoEnricher
from pydantic import Field, create_model

from exps.judge.engines.common import (
    JudgeResult,
    grade_value,
    normalize_rubric,
    rubric_text,
)


def _openai_model_name(model: str) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("strategy.params.model must be a non-empty string.")
    provider, separator, model_name = model.strip().partition("/")
    if separator:
        if provider.lower() != "openai" or not model_name.strip():
            raise ValueError("LLMJudgeEngine requires an OpenAI model.")
        return f"openai/{model_name.strip()}"
    return f"openai/{model.strip()}"


class LLMJudgeEngine:
    """Use AutoEnricher to produce a structured numeric relevance grade."""

    def __init__(
        self,
        *,
        model: str,
        system_prompt: str,
        rubric: Mapping,
        no_cache: bool = False,
    ):
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("strategy.params.system_prompt must be a non-empty string.")
        self.model = _openai_model_name(model)
        self.rubric = normalize_rubric(rubric)
        self.no_cache = no_cache
        grade_literal = Literal.__getitem__(tuple(sorted(self.rubric)))
        self.response_model = create_model(
            "LLMJudgeGrade",
            grade=(
                grade_literal,
                Field(description="The single numeric grade selected from the rubric."),
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
        rubric: Mapping,
        prompt_has_question: bool,
        prompt_has_rubric: bool,
    ) -> JudgeResult:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Judge prompt must be a non-empty string.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Judge question must be a non-empty string.")
        active_rubric = normalize_rubric(rubric)
        if active_rubric != self.rubric:
            raise ValueError("Judge request rubric differs from the configured rubric.")
        parts = [prompt]
        if not prompt_has_question:
            parts.append(question)
        if not prompt_has_rubric:
            parts.append(rubric_text(self.rubric))
        request = "\n\n".join(part for part in parts if part)
        if self.no_cache:
            response = self.enricher.enricher.enrich(request)
        else:
            response = self.enricher.enrich(request)
        grade = getattr(response, "grade", None) if response is not None else None
        if grade is None:
            raise ValueError("Judge response did not include a grade.")
        numeric_grade = grade_value(grade)
        if numeric_grade not in self.rubric:
            raise ValueError(f"Judge returned a grade not in the rubric: {grade!r}.")
        return JudgeResult(grade=numeric_grade)
