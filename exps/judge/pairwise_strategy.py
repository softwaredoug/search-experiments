from __future__ import annotations

import hashlib
import json
import math
import string
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd
from cheat_at_search.strategy import SearchStrategy

from exps.judge.engines.common import PairwiseJudgeResult
from exps.judge.engines.jev_pairwise import JevPairwiseJudgeEngine
from exps.judge.engines.llm_pairwise import LLMPairwiseJudgeEngine
from exps.judge.sample import sample_judgments
from exps.judge.tournament import NoOpScheduler


_OUTCOMES = {"lhs", "rhs", "neither"}


@dataclass(frozen=True)
class PairwiseRequest:
    prompt: str
    question: str
    prompt_has_question: bool


def _template_fields(template: str) -> set[str]:
    try:
        return {
            field_name.split(".", 1)[0].split("[", 1)[0]
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        }
    except ValueError as exc:
        raise ValueError("Judge prompt and question must be valid format strings.") from exc


class PairwiseJudgeStrategy(SearchStrategy):
    """Sample judged documents, compare them pairwise, and Copeland-rank them."""

    _type = "llm_pairwise_judge"

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        judgments: pd.DataFrame | None = None,
        no_cache: bool = False,
        **_kwargs,
    ):
        if judgments is None:
            raise ValueError("The pairwise judge strategy requires dataset judgments.")
        return cls(
            corpus,
            judgments=judgments,
            workers=workers,
            no_cache=no_cache,
            **params,
        )

    def __init__(
        self,
        corpus,
        *,
        judgments: pd.DataFrame,
        model: str = "gpt-5-mini",
        prompt: str,
        question: str,
        to_pointwise: str = "copeland",
        system_prompt: str = "You are a careful search relevance judge.",
        sample_size: int = 10,
        sampling_seed: int = 42,
        confidence_threshold: float | None = None,
        explicit_neither: bool = False,
        double_check: bool = True,
        workers: int = 1,
        no_cache: bool = False,
    ):
        for field, value in (("prompt", prompt), ("question", question)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"strategy.params.{field} must be a non-empty string.")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("strategy.params.system_prompt must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("strategy.params.model must be a non-empty string.")
        if to_pointwise != "copeland":
            raise ValueError("strategy.params.to_pointwise must be 'copeland'.")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise ValueError("strategy.params.sample_size must be an integer from 1 to 10.")
        if not 1 <= sample_size <= 10:
            raise ValueError("strategy.params.sample_size must be an integer from 1 to 10.")
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
            raise ValueError("strategy.params.sampling_seed must be an integer.")
        if not isinstance(explicit_neither, bool):
            raise ValueError("strategy.params.explicit_neither must be a boolean.")
        if not isinstance(double_check, bool):
            raise ValueError("strategy.params.double_check must be a boolean.")
        if confidence_threshold is not None:
            if (
                isinstance(confidence_threshold, bool)
                or not isinstance(confidence_threshold, (int, float))
                or not 0 <= confidence_threshold <= 1
            ):
                raise ValueError(
                    "strategy.params.confidence_threshold must be a number from 0 to 1."
                )

        provider, separator, model_name = model.strip().partition("/")
        provider = provider.lower()
        if separator and (provider not in {"openai", "jev"} or not model_name.strip()):
            raise ValueError(
                "strategy.params.model must be an OpenAI model or a Jev model (jev/*)."
            )
        is_jev = provider == "jev"
        if confidence_threshold is not None and not is_jev:
            raise ValueError(
                "strategy.params.confidence_threshold requires a Jev model (jev/*); "
                "OpenAI pairwise judges do not provide choice probabilities."
            )

        if "query" not in judgments or "doc_id" not in judgments:
            raise ValueError("Judgments must contain 'query' and 'doc_id' columns.")
        grade_col = next(
            (
                column
                for column in ("grade", "relevance", "rel", "label", "score")
                if column in judgments.columns
            ),
            None,
        )
        if grade_col is None:
            raise ValueError("Judgments must contain a numeric grade column.")
        if "doc_id" not in corpus:
            raise ValueError("The corpus must contain a 'doc_id' column.")

        self.corpus = corpus
        self.model = model.strip()
        self.is_jev = is_jev
        self.prompt_template = prompt
        self.question_template = question
        self.system_prompt = system_prompt
        self.to_pointwise = to_pointwise
        self.sample_size = sample_size
        self.sampling_seed = sampling_seed
        self.confidence_threshold = (
            float(confidence_threshold) if confidence_threshold is not None else None
        )
        self.explicit_neither = explicit_neither
        self.double_check = double_check
        self.no_cache = no_cache
        self.grade_column = grade_col
        self._prompt_fields = _template_fields(prompt)
        self._question_fields = _template_fields(question)
        pair_fields = {
            f"{column}_{side}" for column in corpus.columns for side in ("lhs", "rhs")
        }
        missing_fields = sorted(
            (self._prompt_fields - (pair_fields | {"query", "question"}))
            | (self._question_fields - (pair_fields | {"query"}))
        )
        if missing_fields:
            raise ValueError(
                "Pairwise judge placeholders must be 'query', 'question', or corpus "
                "columns suffixed with '_lhs' or '_rhs'; missing: "
                + ", ".join(missing_fields)
            )

        self._samples = sample_judgments(
            corpus,
            judgments,
            grade_column=self.grade_column,
            sample_size=self.sample_size,
            seed=self.sampling_seed,
        )
        self._scheduler_class = NoOpScheduler
        self._rounds = 1
        if is_jev:
            self.engine = JevPairwiseJudgeEngine(
                model=self.model,
                explicit_neither=explicit_neither,
                no_cache=no_cache,
            )
        else:
            self.engine = LLMPairwiseJudgeEngine(
                model=self.model,
                system_prompt=self.system_prompt,
                explicit_neither=explicit_neither,
                no_cache=no_cache,
            )
        super().__init__(corpus, workers=workers)

    def _render_request(self, query: str, lhs_index: int, rhs_index: int) -> PairwiseRequest:
        lhs = self.corpus.iloc[lhs_index]
        rhs = self.corpus.iloc[rhs_index]
        values: dict[str, Any] = {"query": query}
        values.update({f"{key}_lhs": value for key, value in lhs.items()})
        values.update({f"{key}_rhs": value for key, value in rhs.items()})
        try:
            question = self.question_template.format_map(values)
            values["question"] = question
            prompt = self.prompt_template.format_map(values)
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(
                "Pairwise judge placeholders must be 'query', 'question', or corpus "
                "columns suffixed with '_lhs' or '_rhs'."
            ) from exc
        return PairwiseRequest(
            prompt=prompt,
            question=question,
            prompt_has_question="question" in self._prompt_fields,
        )

    @staticmethod
    def _validated_outcome(result) -> PairwiseJudgeResult:
        outcome = getattr(result, "outcome", result)
        if outcome not in _OUTCOMES:
            raise ValueError(f"Pairwise judge returned an invalid outcome: {outcome!r}.")
        probability = getattr(result, "probability", None)
        if probability is not None:
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Pairwise judge returned an invalid choice probability.")
            probability = float(probability)
        return PairwiseJudgeResult(outcome=outcome, probability=probability)

    def _judge_once(self, query: str, lhs_index: int, rhs_index: int) -> PairwiseJudgeResult:
        request = self._render_request(query, lhs_index, rhs_index)
        if self.is_jev:
            result = self.engine.judge(
                state=request.prompt,
                question=request.question,
            )
        else:
            result = self.engine.judge(
                prompt=request.prompt,
                question=request.question,
                prompt_has_question=request.prompt_has_question,
            )
        decision = self._validated_outcome(result)
        if self.confidence_threshold is not None:
            if (
                decision.probability is None
                or decision.probability <= self.confidence_threshold
            ):
                return PairwiseJudgeResult(outcome="neither")
        return decision

    def _compare(self, query: str, lhs_index: int, rhs_index: int) -> str:
        first = self._judge_once(query, lhs_index, rhs_index).outcome
        if not self.double_check:
            return first
        swapped = self._judge_once(query, rhs_index, lhs_index).outcome
        reverse = {"lhs": "rhs", "rhs": "lhs", "neither": "neither"}
        return first if reverse[first] == swapped else "neither"

    def search(self, query: str, k: int = 10):
        sampled = self._samples.get(str(query), [])[:k]
        scheduler = self._scheduler_class(sampled)
        points = {index: 0.0 for index, _ in sampled}
        for round_no in range(self._rounds):
            scheduled = scheduler.schedule(num_nodes=k)
            for (lhs_index, _), (rhs_index, _) in combinations(scheduled, 2):
                outcome = self._compare(str(query), lhs_index, rhs_index)
                if outcome == "lhs":
                    points[lhs_index] += 1.0
                    scheduler.add_preference(lhs_index, rhs_index)
                elif outcome == "rhs":
                    points[rhs_index] += 1.0
                    scheduler.add_preference(rhs_index, lhs_index)
                else:
                    points[lhs_index] += 0.5
                    points[rhs_index] += 0.5
        ranked = sorted(points.items(), key=lambda item: (-item[1], item[0]))
        return [index for index, _ in ranked], [score for _, score in ranked]

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "prompt": self.prompt_template,
            "question": self.question_template,
            "system_prompt": self.system_prompt,
            "to_pointwise": self.to_pointwise,
            "sample_size": self.sample_size,
            "sampling_seed": self.sampling_seed,
            "confidence_threshold": self.confidence_threshold,
            "explicit_neither": self.explicit_neither,
            "double_check": self.double_check,
            "samples": self._samples,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
