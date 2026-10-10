from __future__ import annotations

import math
import string
from collections.abc import Mapping
from typing import Any

from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Choice, TypeSafeClient


_RELEVANT_LABEL = "Relevant"
_NOT_RELEVANT_LABEL = "Not Relevant"
_RELEVANCE_CRITERIA = {
    _RELEVANT_LABEL: "The document satisfies the user's search intent.",
    _NOT_RELEVANT_LABEL: "The document does not satisfy the user's search intent.",
}


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number.")
    return number


def _jev_model_name(model: Any) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ValueError("reranker.params.decision_model must be a Jev model.")
    provider, separator, model_name = model.strip().partition("/")
    if provider.lower() != "jev":
        raise ValueError("reranker.params.decision_model must be a Jev model (jev/*).")
    if not separator:
        return "jev-latest"
    if not model_name.strip():
        raise ValueError("reranker.params.decision_model must include a model name.")
    return model_name.strip()


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        integer = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a positive integer.") from exc
    if integer <= 0 or str(value).strip() not in {str(integer), f"{integer}.0"}:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


def _template_fields(template: str, *, name: str) -> set[str]:
    try:
        return {
            field_name.split(".", 1)[0].split("[", 1)[0]
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        }
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid format string.") from exc


class JevReranker:
    """Rerank a shortlist with TypeSafe relevance choices."""

    def __init__(
        self,
        *,
        corpus,
        k: int,
        decision_model: str,
        decision_weight: float,
        confidence_threshold: float,
        state_format: str,
        prompt: str,
    ):
        self.k = _positive_integer(k, "reranker.k")
        if self.k > 100:
            raise ValueError("reranker.k must be <= 100.")
        self.decision_model = _jev_model_name(decision_model)
        self.decision_weight = _finite_number(
            decision_weight, "reranker.params.decision_weight"
        )
        if self.decision_weight < 0:
            raise ValueError("reranker.params.decision_weight must be non-negative.")
        self.confidence_threshold = _finite_number(
            confidence_threshold, "reranker.params.confidence_threshold"
        )
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                "reranker.params.confidence_threshold must be between 0 and 1."
            )
        if not isinstance(state_format, str) or not state_format.strip():
            raise ValueError("reranker.params.state_format must be a non-empty string.")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("reranker.params.prompt must be a non-empty string.")

        available_fields = set(corpus.columns) | {"id", "query"}
        missing_state_fields = sorted(
            _template_fields(state_format, name="reranker.params.state_format")
            - available_fields
        )
        if missing_state_fields:
            raise ValueError(
                "reranker.params.state_format references missing corpus fields: "
                + ", ".join(missing_state_fields)
            )
        prompt_fields = _template_fields(prompt, name="reranker.params.prompt")
        if prompt_fields - {"query"}:
            raise ValueError(
                "reranker.params.prompt may only use the {query} placeholder."
            )

        if "doc_id" in corpus.columns:
            self._documents = {
                str(row["doc_id"]): row.to_dict()
                for _, row in corpus.iterrows()
            }
        else:
            self._documents = {
                str(index): row.to_dict() for index, row in corpus.iterrows()
            }
        self.state_format = state_format
        self.prompt = prompt
        self.client = TypeSafeClient(
            api_key=key_for_provider("typesafe"),
            model=self.decision_model,
        )

    def _score_candidate(self, query: str, candidate: Mapping[str, Any]) -> float | None:
        doc_id = candidate.get("id", candidate.get("doc_id"))
        document = self._documents.get(str(doc_id), {})
        values = {**document, **candidate, "query": query}
        try:
            state = self.state_format.format_map(values)
            instructions = self.prompt.format_map({"query": query})
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("Could not render the Jev reranker prompt or state.") from exc

        response = self.client.system_one(
            state=state,
            questions={
                "relevance": Choice(
                    instructions=instructions,
                    criteria=_RELEVANCE_CRITERIA,
                )
            },
        )
        answers = getattr(response, "answers", None)
        answer = answers.get("relevance") if isinstance(answers, dict) else None
        confidence = getattr(answer, "confidence", None)
        probabilities = getattr(answer, "probabilities", None)
        probability = (
            probabilities.get(_RELEVANT_LABEL)
            if isinstance(probabilities, dict)
            else None
        )
        if not self._valid_probability(confidence) or not self._valid_probability(probability):
            return None
        if confidence < self.confidence_threshold:
            return None
        return float(probability)

    @staticmethod
    def _valid_probability(value: Any) -> bool:
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 <= value <= 1
        )

    def rerank(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        agent_state: dict | None = None,
    ) -> list[dict[str, Any]]:
        del agent_state  # Reserved for future request tracing/cache context.
        rerank_count = min(self.k, len(candidates))
        reranked = [dict(candidate) for candidate in candidates[:rerank_count]]
        for candidate in reranked:
            relevance_probability = self._score_candidate(query, candidate)
            if relevance_probability is not None:
                candidate["score"] = float(candidate.get("score", 0.0)) + (
                    self.decision_weight * relevance_probability
                )
        reranked.sort(key=lambda candidate: candidate.get("score", 0.0), reverse=True)
        return reranked + [dict(candidate) for candidate in candidates[rerank_count:]]


def make_reranker(corpus, config: dict | None) -> JevReranker | None:
    """Build the configured reranker, or return ``None`` when omitted."""
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError("reranker_engine must be a mapping.")
    if config.get("type") != "jev":
        raise ValueError("reranker_engine currently supports only type 'jev'.")

    params = config.get("params") or {}
    if not isinstance(params, dict):
        raise ValueError("reranker_engine.params must be a mapping.")
    return JevReranker(
        corpus=corpus,
        k=config.get("k", 100),
        decision_model=params.get("decision_model"),
        decision_weight=params.get("decision_weight", 10),
        confidence_threshold=params.get("confidence_threshold", 0.7),
        state_format=params.get("state_format", "{title}\n{description}"),
        prompt=params.get("prompt", "Is this document relevant to the {query}?"),
    )
