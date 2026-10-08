from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np


class FakeEmbeddingModel:
    def __init__(self, vectors: dict[str, list[float]]):
        self.vectors = {
            text: np.asarray(vector, dtype=float) for text, vector in vectors.items()
        }
        self.calls: list[object] = []

    def encode(self, inputs: str | Iterable[str], **_kwargs):
        self.calls.append(inputs)
        if isinstance(inputs, str):
            return self.vectors[inputs]
        values = [self.vectors[text] for text in inputs]
        if not values:
            return np.empty((0, 0), dtype=float)
        return np.vstack(values)


class ScriptedAutoEnricher:
    responses: list[list[str]] = []
    instances: list["ScriptedAutoEnricher"] = []

    def __init__(self, **kwargs):
        self.model = kwargs["model"]
        self.system_prompt = kwargs["system_prompt"]
        self.response_model = kwargs["response_model"]
        self.calls: list[str] = []
        self.raw_calls: list[str] = []
        self.enricher = _ScriptedProviderEnricher(self)
        self.__class__.instances.append(self)

    def enrich(self, prompt: str):
        self.calls.append(prompt)
        return self._response()

    def _response(self):
        values = self.responses.pop(0)
        response_field = next(iter(self.response_model.model_fields))
        return self.response_model(**{response_field: values})


class _ScriptedProviderEnricher:
    def __init__(self, parent: ScriptedAutoEnricher):
        self.parent = parent

    def enrich(self, prompt: str):
        self.parent.raw_calls.append(prompt)
        return self.parent._response()


def patch_hallucinate_dependencies(monkeypatch, *, vectors, responses):
    """Mock embedding and LLM boundaries used by the planned enricher."""
    from cheat_at_search import embeddings
    from cheat_at_search.enrich import enrich as enrich_module

    model = FakeEmbeddingModel(vectors)
    monkeypatch.setattr(embeddings, "_MODEL_REGISTRY", {})
    monkeypatch.setattr(
        embeddings,
        "_load_model",
        lambda _model_name, device=None: model,
    )
    monkeypatch.setattr(enrich_module, "AutoEnricher", ScriptedAutoEnricher)

    ScriptedAutoEnricher.responses = list(responses)
    ScriptedAutoEnricher.instances = []

    module_name = "exps.query_understanding.enrichers.hallucinate_then_resolve"
    try:
        engine_module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
    else:
        if hasattr(engine_module, "load_model"):
            monkeypatch.setattr(
                engine_module,
                "load_model",
                lambda _model_name, device=None: model,
            )
        if hasattr(engine_module, "AutoEnricher"):
            monkeypatch.setattr(
                engine_module, "AutoEnricher", ScriptedAutoEnricher
            )

    return model, ScriptedAutoEnricher


def oracle_test_params(**overrides):
    params = {
        "model": "openai/test-model",
        "resolve_model": "test/resolve-model",
        "system_prompt": "Generate hypothetical product categories.",
        "prompt": (
            "Query: {query}\n"
            "{category_name} may look like:\n"
            "Samples:\n{samples}"
        ),
        "num_samples": 2,
        "max_sample_sim": 0.95,
        "similarity_threshold": 0.75,
    }
    params.update(overrides)
    return {"type": "hallucinate_then_resolve", "params": params}
