from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import string
from typing import Any, Mapping, Sequence

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Noul, NoulCriteria, TypeSafeClient

from exps.bag_of_decisions.decision_question import DecisionQuestion


def _jev_model_name(model: str) -> str:
    provider, separator, model_name = model.partition("/")
    if model == "jev":
        return "jev-latest"
    if not separator or provider.lower() != "jev" or not model_name:
        raise ValueError(
            "decision_engine.decision_model requires a Jev model (jev/*); "
            f"received {model!r}."
        )
    return model_name


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


@dataclass(frozen=True)
class ScoredCandidate:
    """A retrieved document and its current retrieval score."""

    index: int
    score: float
    document: Mapping[str, Any]


class DecisionReranker:
    """Score retrieved documents with Noul decisions and rerank them."""

    def __init__(
        self,
        *,
        decision_model: str,
        decision_weight: float,
        confidence_threshold: float,
        state_format: str,
        corpus_fields: Sequence[str],
        no_cache: bool = False,
    ):
        if not isinstance(decision_model, str) or not decision_model.strip():
            raise ValueError(
                "decision_engine.decision_model must be a non-empty Jev model."
            )
        self.decision_model = _jev_model_name(decision_model.strip())
        self.decision_weight = _finite_number(
            decision_weight, "decision_engine.decision_weight"
        )
        if self.decision_weight < 0:
            raise ValueError("decision_engine.decision_weight must be non-negative.")
        self.confidence_threshold = _finite_number(
            confidence_threshold, "decision_engine.confidence_threshold"
        )
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                "decision_engine.confidence_threshold must be between 0 and 1."
            )
        if not isinstance(state_format, str) or not state_format.strip():
            raise ValueError("decision_engine.state_format must be a non-empty string.")
        self.state_format = state_format
        self._validate_state_format(corpus_fields)
        self.no_cache = no_cache
        self.client = TypeSafeClient(
            api_key=key_for_provider("typesafe"),
            model=self.decision_model,
        )
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    def _validate_state_format(self, corpus_fields: Sequence[str]) -> None:
        try:
            fields = {
                field_name.split(".", 1)[0].split("[", 1)[0]
                for _, field_name, _, _ in string.Formatter().parse(self.state_format)
                if field_name
            }
        except ValueError as exc:
            raise ValueError("decision_engine.state_format is invalid.") from exc
        missing = sorted(fields - set(corpus_fields) - {"query"})
        if missing:
            raise ValueError(
                "decision_engine.state_format references missing corpus fields: "
                + ", ".join(missing)
            )

    def _state(self, document: Mapping[str, Any], query: str) -> str:
        try:
            return self.state_format.format_map({**document, "query": query})
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "Could not render decision_engine.state_format for a corpus row."
            ) from exc

    def _probability_sum(
        self, state: str, decisions: Sequence[DecisionQuestion]
    ) -> float:
        decision_signature = tuple(
            (
                decision.instructions,
                json.dumps(decision.criteria, sort_keys=True),
            )
            for decision in decisions
        )
        cache_key = (state, decision_signature)
        if not self.no_cache and cache_key in self._cache:
            return self._cache[cache_key]
        questions = {}
        for index, decision in enumerate(decisions):
            criteria = (
                NoulCriteria(**decision.criteria) if decision.criteria is not None else None
            )
            questions[f"decision_{index}"] = Noul(
                instructions=decision.instructions,
                criteria=criteria,
            )
        response = self.client.system_one(state=state, questions=questions)
        answers = getattr(response, "answers", {}) or {}
        probability_sum = 0.0
        for question_id in questions:
            answer = answers.get(question_id)
            probability = getattr(answer, "noul", None)
            if (
                isinstance(probability, (int, float))
                and not isinstance(probability, bool)
                and math.isfinite(probability)
                and probability > self.confidence_threshold
            ):
                probability_sum += float(probability)
        if not self.no_cache:
            self._cache[cache_key] = probability_sum
        return probability_sum

    def rerank(
        self,
        candidates: Sequence[ScoredCandidate],
        decisions: Sequence[DecisionQuestion],
        *,
        query: str,
    ) -> list[ScoredCandidate]:
        reranked: list[ScoredCandidate] = []
        for candidate in candidates:
            score = candidate.score
            if decisions:
                probability_sum = self._probability_sum(
                    self._state(candidate.document, query), decisions
                )
                score += self.decision_weight * probability_sum
            reranked.append(replace(candidate, score=score))
        return sorted(reranked, key=lambda candidate: -candidate.score)

    @property
    def cache_key(self) -> str:
        payload = {
            "type": "decision_reranker",
            "decision_model": self.decision_model,
            "decision_weight": self.decision_weight,
            "confidence_threshold": self.confidence_threshold,
            "state_format": self.state_format,
            "no_cache": self.no_cache,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
