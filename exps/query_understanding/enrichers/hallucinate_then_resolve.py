from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
from cheat_at_search.embeddings import load_model
from cheat_at_search.enrich.enrich import AutoEnricher
from pydantic import BaseModel, Field


class HallucinatedClassifications(BaseModel):
    hallucinated_classifications: list[str] = Field(
        description="Hypothetical classifications that describe the user's query."
    )


def _model_name(model: str) -> str:
    return model if "/" in model else f"openai/{model}"


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"hallucinate_then_resolve requires {name} as a non-empty string.")
    return value.strip()


def _cosine_threshold(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"hallucinate_then_resolve {name} must be between -1 and 1.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"hallucinate_then_resolve {name} must be between -1 and 1."
        ) from exc
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError(
            f"hallucinate_then_resolve {name} must be between -1 and 1."
        )
    return threshold


def _num_samples(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(
            "hallucinate_then_resolve num_samples must be a positive integer."
        )
    return value


def _normalize_embeddings(embeddings: Any, *, expected_rows: int) -> np.ndarray:
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    if vectors.ndim != 2 or vectors.shape[0] != expected_rows:
        raise ValueError(
            "resolve_model returned embeddings with an unexpected shape: "
            f"{vectors.shape}; expected {expected_rows} rows."
        )
    if not np.isfinite(vectors).all():
        raise ValueError("resolve_model returned non-finite embeddings.")
    norms = np.linalg.norm(vectors, axis=1)
    norms[norms == 0] = 1.0
    return vectors / norms[:, None]


class HallucinateThenResolveEnricher:
    """Generate hypothetical category names and resolve them against the corpus."""

    def __init__(
        self,
        *,
        field: str,
        vocabulary: list[str],
        model: str,
        system_prompt: str,
        prompt: str,
        num_samples: int,
        max_sample_sim: float,
        resolve_model: str,
        similarity_threshold: float,
        use_cache: bool = True,
        device: str | None = None,
        reasoning: str | None = None,
        temperature: float | None = None,
        verbosity: str | None = None,
    ):
        self.field = _non_empty_string(field, "field")
        self.model = _model_name(_non_empty_string(model, "params.model"))
        self.resolve_model_name = _non_empty_string(
            resolve_model, "params.resolve_model"
        )
        self.system_prompt = _non_empty_string(system_prompt, "params.system_prompt")
        self.prompt_template = _non_empty_string(prompt, "params.prompt")
        self.num_samples = _num_samples(num_samples)
        self.max_sample_sim = _cosine_threshold(max_sample_sim, "max_sample_sim")
        self.similarity_threshold = _cosine_threshold(
            similarity_threshold, "similarity_threshold"
        )
        if not isinstance(use_cache, bool):
            raise ValueError("hallucinate_then_resolve cache must be a boolean.")
        self.use_cache = use_cache
        self.device = device
        self.reasoning = reasoning
        self.temperature = temperature
        self.verbosity = verbosity
        self.vocabulary = list(
            dict.fromkeys(
                value.strip()
                for value in vocabulary
                if isinstance(value, str) and value.strip()
            )
        )
        self._cache: dict[str, list[str]] = {}
        self.samples: list[str] = []

        if not self.vocabulary:
            self.category_embeddings = np.empty((0, 0), dtype=float)
            self.embedding_model = None
            self.enricher = None
            return

        self.embedding_model = load_model(self.resolve_model_name, device=device)
        self.category_embeddings = _normalize_embeddings(
            self.embedding_model.encode(self.vocabulary, convert_to_numpy=True),
            expected_rows=len(self.vocabulary),
        )
        self.samples = self._select_samples()
        self._samples_prompt = "\n".join(f"- {sample}" for sample in self.samples)
        self.enricher = AutoEnricher(
            model=self.model,
            system_prompt=self.system_prompt,
            response_model=HallucinatedClassifications,
            temperature=self.temperature,
            reasoning_effort=self.reasoning,
            verbosity=self.verbosity,
        )

    def _select_samples(self) -> list[str]:
        selected: list[int] = []
        for candidate_idx, vector in enumerate(self.category_embeddings):
            if selected:
                similarities = self.category_embeddings[selected] @ vector
                if np.any(similarities >= self.max_sample_sim):
                    continue
            selected.append(candidate_idx)
            if len(selected) == self.num_samples:
                break
        return [self.vocabulary[index] for index in selected]

    def _prompt(self, query: str) -> str:
        try:
            prompt = self.prompt_template.format(
                query=query,
                category_name=self.field,
                samples=self._samples_prompt,
            )
            print("hallucinate_then_resolve prompt:", prompt)
            return prompt
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "hallucinate_then_resolve prompt may use {query}, "
                "{category_name}, and {samples}."
            ) from exc

    def _resolve(self, hallucinations: list[str]) -> list[str]:
        if not hallucinations:
            return []
        hypothesis_embeddings = _normalize_embeddings(
            self.embedding_model.encode(hallucinations, convert_to_numpy=True),
            expected_rows=len(hallucinations),
        )

        resolved: list[str] = []
        resolved_indices: set[int] = set()
        for vector, hallucination in zip(hypothesis_embeddings, hallucinations):
            similarities = self.category_embeddings @ vector
            if resolved_indices:
                similarities[list(resolved_indices)] = -np.inf
            best_idx = int(np.argmax(similarities))
            best_similarity = float(similarities[best_idx])
            # Keep the threshold strict while avoiding floating-point equality drift.
            if best_similarity > self.similarity_threshold + 1e-12:
                print(hallucination, "->", self.vocabulary[best_idx], f"(similarity={best_similarity:.4f})")
                resolved.append(self.vocabulary[best_idx])
                resolved_indices.add(best_idx)
            else:
                best_match = self.vocabulary[best_idx] if best_similarity > -np.inf else "<no match>"
                print(hallucination, "-/->", best_match, f"(similarity={best_similarity:.4f})")
        return resolved

    def enrich(self, query: str) -> list[str]:
        if self.use_cache and query in self._cache:
            return list(self._cache[query])
        if not self.vocabulary:
            if self.use_cache:
                self._cache[query] = []
            return []

        prompt = self._prompt(query)
        if self.use_cache:
            response = self.enricher.enrich(prompt)
        else:
            # AutoEnricher.enrich uses its persistent CachedEnrichClient. Call
            # the wrapped provider client directly when caching is disabled.
            response = self.enricher.enricher.enrich(prompt)
        hallucinations = (
            getattr(response, "hallucinated_classifications", None)
            if response is not None
            else None
        )
        if not isinstance(hallucinations, (list, tuple)):
            hallucinations = []
        hallucinations = [
            value.strip()
            for value in hallucinations
            if isinstance(value, str) and value.strip()
        ]
        categories = self._resolve(hallucinations)
        if self.use_cache:
            self._cache[query] = categories
        return list(categories)

    @property
    def cache_key(self) -> str:
        payload = {
            "type": "hallucinate_then_resolve",
            "field": self.field,
            "vocabulary": self.vocabulary,
            "samples": self.samples,
            "model": self.model,
            "resolve_model": self.resolve_model_name,
            "similarity_threshold": self.similarity_threshold,
            "num_samples": self.num_samples,
            "max_sample_sim": self.max_sample_sim,
            "system_prompt": self.system_prompt,
            "prompt": self.prompt_template,
            "reasoning": self.reasoning,
            "temperature": self.temperature,
            "verbosity": self.verbosity,
            "device": self.device,
            "use_cache": self.use_cache,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()


def make_hallucinate_then_resolve_enricher(
    *,
    field: str,
    vocabulary: list[str],
    params: dict[str, Any],
    model: str = "gpt-5-mini",
    reasoning: str | None = None,
    device: str | None = None,
    no_cache: bool = False,
) -> HallucinateThenResolveEnricher:
    use_cache = params.get("cache", True)
    if not isinstance(use_cache, bool):
        raise ValueError("hallucinate_then_resolve params.cache must be a boolean.")
    return HallucinateThenResolveEnricher(
        field=field,
        vocabulary=vocabulary,
        model=params.get("model", model),
        system_prompt=params.get("system_prompt"),
        prompt=params.get("prompt"),
        num_samples=params.get("num_samples"),
        max_sample_sim=params.get("max_sample_sim"),
        resolve_model=params.get("resolve_model"),
        similarity_threshold=params.get("similarity_threshold"),
        use_cache=use_cache and not no_cache,
        device=device,
        reasoning=params.get("reasoning", reasoning),
        temperature=params.get("temperature"),
        verbosity=params.get("verbosity"),
    )
