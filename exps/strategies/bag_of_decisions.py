from __future__ import annotations

import hashlib
import json
import math
import string
from typing import Any

import numpy as np
from cheat_at_search.enrich.enrich import AutoEnricher
from cheat_at_search.strategy import SearchStrategy
from cheat_at_search.tokenizers import snowball_tokenizer
from cheat_at_search.data_dir import key_for_provider
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity
from pydantic import Field, create_model
from typesafe_sdk import Noul, TypeSafeClient


def _llm_model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


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


def _parse_fields(fields: Any) -> dict[str, float]:
    if not isinstance(fields, (list, tuple)):
        raise ValueError("retrieval_engine.params.fields must be a list of fields.")
    parsed: dict[str, float] = {}
    for field_spec in fields:
        if not isinstance(field_spec, str) or not field_spec.strip():
            raise ValueError("retrieval_engine.params.fields must contain strings.")
        field, separator, weight = field_spec.rpartition("^")
        if not separator:
            field, weight = field_spec, "1.0"
        field = field.strip()
        if not field:
            raise ValueError(f"Invalid BM25 field specification: {field_spec!r}")
        try:
            parsed_weight = float(weight)
        except ValueError as exc:
            raise ValueError(f"Invalid BM25 field weight: {field_spec!r}") from exc
        if not math.isfinite(parsed_weight) or parsed_weight <= 0:
            raise ValueError(f"BM25 field weight must be positive: {field_spec!r}")
        if field in parsed:
            raise ValueError(f"Duplicate BM25 field: {field}")
        parsed[field] = parsed_weight
    if not parsed:
        raise ValueError("retrieval_engine.params.fields must not be empty.")
    return parsed


class BagOfDecisionsStrategy(SearchStrategy):
    """BM25 retrieval followed by query-generated Jev decision reranking."""

    _type = "bag_of_decisions"

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        no_cache: bool = False,
        **kwargs,
    ):
        return cls(
            corpus,
            workers=workers,
            no_cache=no_cache,
            **params,
        )

    def __init__(
        self,
        corpus,
        decision_engine: dict,
        retrieval_engine: dict,
        workers: int = 1,
        top_k: int = 10,
        no_cache: bool = False,
        **_unused,
    ):
        super().__init__(corpus, top_k=top_k, workers=workers)
        self.corpus = corpus
        self.no_cache = no_cache

        if not isinstance(decision_engine, dict):
            raise ValueError("decision_engine must be a mapping.")
        self.system_prompt = decision_engine.get("system_prompt")
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("decision_engine.system_prompt must be a non-empty string.")
        self.prompt_template = decision_engine.get("prompt")
        if not isinstance(self.prompt_template, str) or not self.prompt_template.strip():
            raise ValueError("decision_engine.prompt must be a non-empty string.")

        model = decision_engine.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("decision_engine.model must be a non-empty string.")
        self.model = _llm_model_name(model.strip())
        decision_model = decision_engine.get("decision_model")
        if not isinstance(decision_model, str) or not decision_model.strip():
            raise ValueError(
                "decision_engine.decision_model must be a non-empty Jev model."
            )
        self.decision_model = _jev_model_name(decision_model.strip())

        self.decision_weight = _finite_number(
            decision_engine.get("decision_weight", 10),
            "decision_engine.decision_weight",
        )
        if self.decision_weight < 0:
            raise ValueError("decision_engine.decision_weight must be non-negative.")
        self.confidence_threshold = _finite_number(
            decision_engine.get("confidence_threshold", 0.7),
            "decision_engine.confidence_threshold",
        )
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError(
                "decision_engine.confidence_threshold must be between 0 and 1."
            )
        candidate_k = decision_engine.get("k", 100)
        if isinstance(candidate_k, bool) or not isinstance(candidate_k, int) or candidate_k < 1:
            raise ValueError("decision_engine.k must be a positive integer.")
        self.candidate_k = candidate_k

        self.state_format = decision_engine.get("state_format")
        if not isinstance(self.state_format, str) or not self.state_format.strip():
            raise ValueError("decision_engine.state_format must be a non-empty string.")
        self._validate_state_format()

        self.reasoning = decision_engine.get("reasoning")
        self.temperature = decision_engine.get("temperature")
        self.verbosity = decision_engine.get("verbosity")
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
        self.question_generator = AutoEnricher(
            model=self.model,
            system_prompt=self.system_prompt,
            response_model=self.response_model,
            temperature=self.temperature,
            reasoning_effort=self.reasoning,
            verbosity=self.verbosity,
        )
        self._question_cache: dict[str, list[str]] = {}
        self._decision_cache: dict[tuple[str, str, tuple[str, ...]], float] = {}

        if not isinstance(retrieval_engine, dict):
            raise ValueError("retrieval_engine must be a mapping.")
        retrieval_base = retrieval_engine.get("base", "bm25_boosted")
        supported_bases = {
            "bm25_boosted",
            "bm25_filtered",
            "bm25_hierarchy_boosted",
        }
        if retrieval_base not in supported_bases:
            raise ValueError(
                "retrieval_engine.base must be one of: "
                + ", ".join(sorted(supported_bases))
                + "."
            )
        retrieval_params = retrieval_engine.get("params") or {}
        if not isinstance(retrieval_params, dict):
            raise ValueError("retrieval_engine.params must be a mapping.")
        self.fields = _parse_fields(retrieval_params.get("fields") or [])
        for field in self.fields:
            if field not in corpus.columns:
                raise ValueError(f"Missing BM25 field: {field}")
            index_name = f"{field}_snowball"
            if index_name not in corpus:
                corpus[index_name] = SearchArray.index(
                    corpus[field].fillna("").astype(str), snowball_tokenizer
                )
        self.k1 = _finite_number(retrieval_params.get("k1", 1.2), "k1")
        self.b = _finite_number(retrieval_params.get("b", 0.75), "b")
        if self.k1 <= 0:
            raise ValueError("retrieval_engine.params.k1 must be positive.")
        if not 0 <= self.b <= 1:
            raise ValueError("retrieval_engine.params.b must be between 0 and 1.")

        self.decision_client = TypeSafeClient(
            api_key=key_for_provider("typesafe"),
            model=self.decision_model,
        )

    def _validate_state_format(self) -> None:
        try:
            fields = {
                field_name.split(".", 1)[0].split("[", 1)[0]
                for _, field_name, _, _ in string.Formatter().parse(self.state_format)
                if field_name
            }
        except ValueError as exc:
            raise ValueError("decision_engine.state_format is invalid.") from exc
        missing = sorted(fields - set(self.corpus.columns))
        if missing:
            raise ValueError(
                "decision_engine.state_format references missing corpus fields: "
                + ", ".join(missing)
            )

    def _baseline_scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.corpus), dtype=float)
        similarity = bm25_similarity(k1=self.k1, b=self.b)
        for term in snowball_tokenizer(query):
            for field, weight in self.fields.items():
                scores += (
                    self.corpus[f"{field}_snowball"].array.score(
                        term, similarity=similarity
                    )
                    * weight
                )
        return scores

    def _questions(self, query: str) -> list[str]:
        if not self.no_cache and query in self._question_cache:
            return list(self._question_cache[query])
        try:
            prompt = self.prompt_template.format(query=query)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("decision_engine.prompt may use {query}.") from exc
        if self.no_cache:
            response = self.question_generator.enricher.enrich(prompt)
        else:
            response = self.question_generator.enrich(prompt)
        decisions = getattr(response, "decisions", None) if response is not None else None
        if not isinstance(decisions, (list, tuple)):
            decisions = []
        normalized: list[str] = []
        for decision in decisions:
            if isinstance(decision, str) and decision.strip() and decision.strip() not in normalized:
                normalized.append(decision.strip())
        if not self.no_cache:
            self._question_cache[query] = normalized
        return list(normalized)

    def _formatted_state(self, document) -> str:
        try:
            return self.state_format.format_map(document.to_dict())
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "Could not render decision_engine.state_format for a corpus row."
            ) from exc

    def _decision_score(
        self, *, query: str, state: str, decisions: list[str]
    ) -> float:
        cache_key = (query, state, tuple(decisions))
        if not self.no_cache and cache_key in self._decision_cache:
            return self._decision_cache[cache_key]
        questions = {
            f"decision_{index}": Noul(instructions=decision)
            for index, decision in enumerate(decisions)
        }
        response = self.decision_client.system_one(state=state, questions=questions)
        answers = getattr(response, "answers", {}) or {}
        probability_sum = 0.0
        for question_id in questions:
            answer = answers.get(question_id)
            probability = getattr(answer, "noul", None)
            if (
                isinstance(probability, (int, float))
                and math.isfinite(probability)
                and probability > self.confidence_threshold
            ):
                probability_sum += float(probability)
        if not self.no_cache:
            self._decision_cache[cache_key] = probability_sum
        return probability_sum

    def search(self, query: str, k: int = 10):
        if k <= 0 or not len(self.corpus):
            return np.asarray([], dtype=int), np.asarray([], dtype=float)
        scores = self._baseline_scores(query)
        candidate_indices = np.argsort(-scores, kind="stable")[: self.candidate_k]
        decisions = self._questions(query)
        if decisions:
            for index in candidate_indices:
                state = self._formatted_state(self.corpus.iloc[index])
                probability_sum = self._decision_score(
                    query=query,
                    state=state,
                    decisions=decisions,
                )
                scores[index] += self.decision_weight * probability_sum
        result_indices = np.argsort(-scores, kind="stable")[:k]
        return result_indices, scores[result_indices]

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt_template,
            "decision_model": self.decision_model,
            "decision_weight": self.decision_weight,
            "confidence_threshold": self.confidence_threshold,
            "candidate_k": self.candidate_k,
            "state_format": self.state_format,
            "fields": self.fields,
            "k1": self.k1,
            "b": self.b,
            "reasoning": self.reasoning,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            "top_k": getattr(self, "top_k", None),
            "no_cache": self.no_cache,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
