from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exps.query_understanding.enrichers import make_enricher
from exps.query_understanding.strategy import QueryUnderstandingStrategy
from tests.utils.hallucinate_then_resolve_mocks import (
    ScriptedAutoEnricher,
    oracle_test_params,
    patch_hallucinate_dependencies,
)


def _make_enricher(monkeypatch, corpus, *, vectors, responses, params=None):
    patch_hallucinate_dependencies(
        monkeypatch,
        vectors=vectors,
        responses=responses,
    )
    config = oracle_test_params()
    if params:
        config["params"].update(params)
    vocabulary = (
        corpus["category"]
        .dropna()
        .astype(str)
        .loc[lambda values: values.str.strip() != ""]
        .unique()
        .tolist()
    )
    return make_enricher(
        config,
        field="category",
        vocabulary=vocabulary,
        corpus=corpus,
    )


def test_hallucinations_resolve_to_multiple_unique_categories_once_per_query(
    monkeypatch,
):
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "category": ["CatFurniture", "CatLighting", "CatOutdoor"],
        }
    )
    vectors = {
        "CatFurniture": [1, 0, 0],
        "CatLighting": [0, 1, 0],
        "CatOutdoor": [0, 0, 1],
        "HypoFurniture": [0.99, 0.01, 0],
        "HypoLighting": [0.01, 0.99, 0],
    }
    enricher = _make_enricher(
        monkeypatch,
        corpus,
        vectors=vectors,
        responses=[
            ["HypoFurniture", "HypoLighting"],
            ["HypoFurniture"],
        ],
    )

    assert enricher.enrich("desk and lamp") == ["CatFurniture", "CatLighting"]
    assert enricher.enrich("office chair") == ["CatFurniture"]

    client = ScriptedAutoEnricher.instances[0]
    assert len(ScriptedAutoEnricher.instances) == 1
    assert len(client.calls) == 2
    assert "Query: desk and lamp" in client.calls[0]
    assert "Query: office chair" in client.calls[1]
    assert "category may look like" in client.calls[0]
    first_samples = client.calls[0].split("Samples:\n", 1)[1]
    second_samples = client.calls[1].split("Samples:\n", 1)[1]
    assert first_samples == second_samples


def test_shared_samples_are_corpus_categories_and_pairwise_diverse(monkeypatch):
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2, 3, 4, 5, 6],
            "category": [
                "CatAlpha",
                "CatAlpha",
                "CatAlpha",
                "CatAlphaSimilar",
                "CatAlphaSimilar",
                "CatBeta",
            ],
        }
    )
    vectors = {
        "CatAlpha": [1, 0, 0],
        "CatAlphaSimilar": [0.995, 0.1, 0],
        "CatBeta": [0, 1, 0],
    }
    enricher = _make_enricher(
        monkeypatch,
        corpus,
        vectors=vectors,
        responses=[[], []],
        params={"max_sample_sim": 0.95},
    )

    assert enricher.enrich("query one") == []
    assert enricher.enrich("query two") == []

    prompts = ScriptedAutoEnricher.instances[0].calls
    sample_texts = [prompt.split("Samples:\n", 1)[1] for prompt in prompts]
    assert sample_texts[0] == sample_texts[1]
    sample_lines = [
        line.strip().lstrip("-*•0123456789.) ").strip()
        for line in sample_texts[0].splitlines()
    ]
    selected = [value for value in vectors if value in sample_lines]
    assert len(selected) == 2
    assert set(selected) <= set(corpus["category"])
    selected_vectors = np.vstack([vectors[value] for value in selected]).astype(float)
    selected_vectors /= np.linalg.norm(selected_vectors, axis=1, keepdims=True)
    pairwise = selected_vectors @ selected_vectors.T
    np.fill_diagonal(pairwise, -1)
    assert np.max(pairwise) < 0.95


def test_sample_selection_uses_available_diverse_values_when_fewer_than_requested(
    monkeypatch,
):
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "category": ["CatAlpha", "CatAlphaSimilar"],
        }
    )
    vectors = {
        "CatAlpha": [1, 0],
        "CatAlphaSimilar": [0.999, 0.04],
    }
    enricher = _make_enricher(
        monkeypatch,
        corpus,
        vectors=vectors,
        responses=[[]],
        params={"num_samples": 3, "max_sample_sim": 0.95},
    )

    assert enricher.enrich("query") == []
    prompt = ScriptedAutoEnricher.instances[0].calls[0]
    sample_lines = [
        line.strip().lstrip("-*•0123456789.) ").strip()
        for line in prompt.split("Samples:\n", 1)[1].splitlines()
    ]
    selected = [value for value in vectors if value in sample_lines]
    assert len(selected) == 1


def test_resolver_uses_strict_threshold_and_does_not_repeat_categories(monkeypatch):
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "category": ["CatFurniture", "CatLighting"],
        }
    )
    vectors = {
        "CatFurniture": [1, 0],
        "CatLighting": [0, 1],
        "AtThreshold": [0.9, float(np.sqrt(1 - 0.9**2))],
        "HypoFurniture": [1, 0],
        "ZeroEmbedding": [0, 0],
    }
    enricher = _make_enricher(
        monkeypatch,
        corpus,
        vectors=vectors,
        responses=[
            ["AtThreshold", "HypoFurniture", "HypoFurniture", "ZeroEmbedding"]
        ],
        params={"similarity_threshold": 0.9},
    )

    assert enricher.enrich("office chair") == ["CatFurniture"]


def test_blank_hallucinations_and_empty_category_vocabulary_return_empty(
    monkeypatch,
):
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "category": [None, "  "],
        }
    )
    model, _ = patch_hallucinate_dependencies(
        monkeypatch,
        vectors={},
        responses=[["", "  "]],
    )
    enricher = make_enricher(
        oracle_test_params(),
        field="category",
        vocabulary=[],
        corpus=corpus,
    )

    assert enricher.enrich("anything") == []
    assert model.calls == []
    assert ScriptedAutoEnricher.instances == []


def test_blank_hallucinated_values_return_empty(monkeypatch):
    corpus = pd.DataFrame({"doc_id": [1], "category": ["CatFurniture"]})
    enricher = _make_enricher(
        monkeypatch,
        corpus,
        vectors={"CatFurniture": [1, 0]},
        responses=[["", "   "]],
    )

    assert enricher.enrich("ambiguous request") == []


@pytest.mark.parametrize(
    ("override", "error_fragment"),
    [
        ({"num_samples": 0}, "num_samples"),
        ({"num_samples": -1}, "num_samples"),
        ({"num_samples": 1.5}, "num_samples"),
        ({"max_sample_sim": 1.01}, "max_sample_sim"),
        ({"max_sample_sim": float("nan")}, "max_sample_sim"),
        ({"similarity_threshold": -1.01}, "similarity_threshold"),
        ({"similarity_threshold": 1.01}, "similarity_threshold"),
        ({"prompt": "  "}, "prompt"),
        ({"system_prompt": "  "}, "system_prompt"),
        ({"resolve_model": "  "}, "resolve_model"),
    ],
)
def test_invalid_hallucinate_then_resolve_params_are_rejected(
    override, error_fragment
):
    corpus = pd.DataFrame({"doc_id": [1], "category": ["CatFurniture"]})
    config = oracle_test_params(**override)

    with pytest.raises(ValueError, match=error_fragment):
        make_enricher(
            config,
            field="category",
            vocabulary=["CatFurniture"],
            corpus=corpus,
        )


def test_query_understanding_build_keeps_all_nonempty_categories_for_engine(
    monkeypatch,
):
    legal_categories = [f"Category-{index}" for index in range(305)]
    corpus = pd.DataFrame(
        {
            "doc_id": list(range(len(legal_categories) + 2)),
            "category": [*legal_categories, "", "   "],
            "title": ["sample"] * (len(legal_categories) + 2),
        }
    )
    captured = {}

    class RecordingEnricher:
        cache_key = "recording-enricher"

        def enrich(self, query):
            return []

    def capture_factory(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return RecordingEnricher()

    monkeypatch.setattr(
        "exps.query_understanding.strategy.make_enricher", capture_factory
    )
    params = {
        "categorize": {
            "field": "category",
            "enrichment_engine": {"type": "hallucinate_then_resolve"},
        },
        "retrieval_engine": {
            "base": "bm25_boosted",
            "params": {"fields": ["title"]},
        },
    }

    QueryUnderstandingStrategy.build(params, corpus=corpus)

    assert len(captured["vocabulary"]) == len(legal_categories)
    assert set(captured["vocabulary"]) == set(legal_categories)
