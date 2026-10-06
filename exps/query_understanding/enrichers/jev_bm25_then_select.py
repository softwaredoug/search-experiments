from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from searcharray import SearchArray
from searcharray.similarity import bm25_similarity

from cheat_at_search.data_dir import DATA_PATH, key_for_provider
from cheat_at_search.tokenizers import snowball_tokenizer
from typesafe_sdk import Choice, TypeSafeClient

from exps.query_understanding.enrichers.choice_single import (
    MAX_JEV_CHOICE_COUNT,
    _choice_confidence_threshold,
    _validate_choice_prompt,
)
from exps.query_understanding.enrichers.choice_single_jev import _model_name


def _parse_bm25_fields(fields: Any) -> dict[str, float]:
    if not isinstance(fields, list) or not fields:
        raise ValueError(
            "jev_bm25_then_select requires params.retrieval.fields as a non-empty list."
        )

    parsed: dict[str, float] = {}
    for field_spec in fields:
        if not isinstance(field_spec, str) or not field_spec.strip():
            raise ValueError("params.retrieval.fields must contain strings.")
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
    return parsed


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"retrieval.{name} must be a positive number.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"retrieval.{name} must be a positive number.")
    return parsed


def _parse_b(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("retrieval.b must be a number between 0 and 1.") from exc
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError("retrieval.b must be a number between 0 and 1.")
    return parsed


class JevBM25ThenSelectEnricher:
    """Retrieve BM25 candidate categories, then select one with Jev Choice."""

    def __init__(
        self,
        *,
        corpus,
        field: str,
        model: str,
        prompt: str,
        aggregate_over: int,
        confidence_threshold: float | None,
        fields: dict[str, float],
        k1: float,
        b: float,
    ):
        self.corpus = corpus
        self.field = field
        self.model = _model_name(model)
        self.prompt_template = prompt
        self.aggregate_over = aggregate_over
        self.confidence_threshold = confidence_threshold
        self.fields = dict(fields)
        self.k1 = k1
        self.b = b

        if self.field not in corpus.columns:
            raise ValueError(f"Missing category field: {self.field}")
        for name in self.fields:
            if name not in corpus.columns:
                raise ValueError(f"Missing BM25 candidate field: {name}")
            index_name = f"{name}_snowball"
            if index_name not in corpus:
                corpus[index_name] = SearchArray.index(
                    corpus[name].fillna(""), snowball_tokenizer
                )

        self.client = TypeSafeClient(
            api_key=key_for_provider("typesafe"),
            model=self.model,
        )
        cache_config = {
            "type": "jev_bm25_then_select",
            "field": self.field,
            "model": self.model,
            "prompt": self.prompt_template,
            "aggregate_over": self.aggregate_over,
            "confidence_threshold": self.confidence_threshold,
            "fields": self.fields,
            "k1": self.k1,
            "b": self.b,
        }
        serialized = json.dumps(cache_config, sort_keys=True).encode("utf-8")
        self.cache_key = hashlib.md5(serialized).hexdigest()
        self.cache_path = (
            Path(DATA_PATH)
            / "query_understanding_cache"
            / f"{self.cache_key}.json"
        )
        self._cache = self._load_cache()

    def _candidate_categories(self, query: str) -> list[str]:
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

        positive_indices = np.flatnonzero(scores > 0)
        if not len(positive_indices):
            return []
        order = np.argsort(-scores[positive_indices])[: self.aggregate_over]
        matched_rows = self.corpus.iloc[positive_indices[order]]
        categories = matched_rows[self.field].dropna().astype(str)
        categories = categories[categories.str.strip() != ""]
        return categories.value_counts().head(MAX_JEV_CHOICE_COUNT).index.tolist()

    def _cache_entry_key(self, query: str, categories: list[str]) -> str:
        payload = json.dumps([query, categories], ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_cache(self) -> dict[str, list[str]]:
        try:
            contents = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
            return {}
        if not isinstance(contents, dict):
            return {}
        return {
            key: value
            for key, value in contents.items()
            if isinstance(key, str)
            and isinstance(value, list)
            and all(isinstance(category, str) for category in value)
        }

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.cache_path.parent,
                prefix=f".{self.cache_key}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                json.dump(self._cache, temporary_file, ensure_ascii=False, sort_keys=True)
                temporary_file.write("\n")
            os.replace(temporary_path, self.cache_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def enrich(self, query: str) -> list[str]:
        categories = self._candidate_categories(query)
        if not categories:
            return []

        cache_entry_key = self._cache_entry_key(query, categories)
        if cache_entry_key in self._cache:
            return list(self._cache[cache_entry_key])

        response = self.client.system_one(
            state=query,
            questions={
                self.field: Choice(
                    instructions=self.prompt_template.format(
                        field=self.field, query=query
                    ),
                    criteria={category: None for category in categories},
                )
            },
        )
        answer = response.choices[self.field]
        selected = getattr(answer, "choice", None)
        confidence = getattr(answer, "confidence", None)
        if (
            self.confidence_threshold is not None
            and (
                not isinstance(confidence, (int, float))
                or not math.isfinite(confidence)
                or confidence <= self.confidence_threshold
            )
        ):
            selected = None
        result = [str(selected)] if selected in categories else []
        self._cache[cache_entry_key] = result
        self._save_cache()
        return list(result)


def make_jev_bm25_then_select_enricher(
    *,
    corpus,
    field: str,
    params: dict[str, Any],
    model: str = "gpt-5-mini",
):
    prompt = params.get("prompt")
    _validate_choice_prompt(prompt, "jev_bm25_then_select")
    configured_model = str(params.get("model", model))
    if configured_model.split("/", 1)[0].lower() != "jev":
        raise ValueError(
            "jev_bm25_then_select requires a Jev model (jev/*); "
            f"received {configured_model!r}."
        )

    aggregate_over = params.get("aggregate_over")
    if (
        isinstance(aggregate_over, bool)
        or not isinstance(aggregate_over, int)
        or aggregate_over <= 0
    ):
        raise ValueError("aggregate_over must be a positive integer.")

    retrieval = params.get("retrieval") or {}
    if not isinstance(retrieval, dict):
        raise ValueError("params.retrieval must be a mapping.")
    fields = _parse_bm25_fields(retrieval.get("fields"))
    k1 = _positive_float(retrieval.get("k1", 1.2), "k1")
    b = _parse_b(retrieval.get("b", 0.75))

    return JevBM25ThenSelectEnricher(
        corpus=corpus,
        field=field,
        model=configured_model,
        prompt=prompt,
        aggregate_over=aggregate_over,
        confidence_threshold=_choice_confidence_threshold(params),
        fields=fields,
        k1=k1,
        b=b,
    )


__all__ = ["JevBM25ThenSelectEnricher", "make_jev_bm25_then_select_enricher"]
