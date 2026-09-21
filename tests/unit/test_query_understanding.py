from unittest.mock import patch

import pandas as pd
import pytest
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer

from exps.query_understanding.enrichers import make_enricher
from exps.query_understanding.enrichers import make_llm_multiple_enricher
from exps.query_understanding.enrichers import make_llm_single_enricher
from exps.query_understanding import QueryUnderstandingStrategy
from exps.query_understanding.strategy import MAX_CATEGORY_CARDINALITY
from exps.runners.query_classification import _ground_truth


class FakeAutoEnricher:
    value = "Furniture"
    instances = []

    def __init__(self, **kwargs):
        self.response_model = kwargs["response_model"]
        self.system_prompt = kwargs["system_prompt"]
        self.prompts = []
        self.__class__.instances.append(self)

    def enrich(self, prompt):
        self.prompts.append(prompt)
        field = next(iter(self.response_model.model_fields))
        return self.response_model(**{field: self.value})


def test_llm_single_enricher_builds_prompt_and_normalizes_response():
    FakeAutoEnricher.instances = []
    with patch(
        "exps.query_understanding.enrichers.llm_single.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_llm_single_enricher(
            field="category",
            vocabulary=["Furniture", "Lighting"],
            prompt="Classify {query} into {field}.",
            model="gpt-5-mini",
            reasoning="medium",
        )
        assert enricher.enrich("sofa") == ["Furniture"]
        assert enricher.enrich("sofa") == ["Furniture"]

    auto_enricher = FakeAutoEnricher.instances[0]
    assert auto_enricher.system_prompt == (
        "You are a helpful furniture shopping agent that helps users construct search queries."
    )
    assert len(auto_enricher.prompts) == 1
    assert "Classify sofa into category." == auto_enricher.prompts[0]
    assert "sofa" in auto_enricher.prompts[0]


def test_llm_single_enricher_maps_unknown_to_empty_list():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = "Unknown"
    try:
        with patch(
            "exps.query_understanding.enrichers.llm_single.AutoEnricher",
            FakeAutoEnricher,
        ):
            enricher = make_llm_single_enricher(
                field="category",
                vocabulary=["Furniture"],
                prompt="Classify {query} into {field}.",
            )
            assert enricher.enrich("ambiguous") == []
    finally:
        FakeAutoEnricher.value = "Furniture"


def test_llm_single_enricher_requires_prompt():
    with pytest.raises(ValueError, match="params.prompt"):
        make_enricher(
            {"type": "llm_single"},
            field="category",
            vocabulary=["Furniture"],
        )


def test_llm_multiple_enricher_returns_deduplicated_categories():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = ["Furniture", "Lighting", "Unknown", "Furniture"]
    try:
        with patch(
            "exps.query_understanding.enrichers.llm_multiple.AutoEnricher",
            FakeAutoEnricher,
        ):
            enricher = make_llm_multiple_enricher(
                field="category",
                vocabulary=["Furniture", "Lighting", "Outdoor"],
                prompt="Classify {query} into {field} values.",
            )
            assert enricher.enrich("sofa") == ["Furniture", "Lighting"]
            assert enricher.enrich("sofa") == ["Furniture", "Lighting"]

        auto_enricher = FakeAutoEnricher.instances[0]
        assert auto_enricher.prompts == ["Classify sofa into category values."]
    finally:
        FakeAutoEnricher.value = "Furniture"


def test_llm_multiple_enricher_requires_prompt():
    with pytest.raises(ValueError, match="params.prompt"):
        make_enricher(
            {"type": "llm_multiple"},
            field="category",
            vocabulary=["Furniture"],
        )


def test_ground_truth_uses_maximum_grade_regardless_of_scale():
    judgments = pd.DataFrame(
        {
            "query": ["sofa", "sofa", "sofa"],
            "doc_id": [1, 2, 3],
            "grade": [100, 50, 0],
        }
    )
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "category": ["Furniture", "Lighting", "Outdoor"],
        }
    )

    truth = _ground_truth(
        queries=["sofa"],
        judgments=judgments,
        corpus=corpus,
        category_field="category",
        threshold=0.8,
    )

    assert truth == {"sofa": ["Furniture"]}


def test_category_matching_uses_phrase_matching():
    corpus = pd.DataFrame(
        {
            "title": ["one", "two", "three", "four"],
            "description": ["", "", "", ""],
            "category": ["Living Room", "Living", "Room", "Dining Room"],
        }
    )
    corpus["category_snowball"] = SearchArray.index(
        corpus["category"], snowball_tokenizer
    )
    strategy = QueryUnderstandingStrategy(
        corpus,
        categorize={"field": "category"},
        retrieval_engine={
            "base": "bm25_boosted",
            "params": {"fields": ["title^1"]},
        },
        enricher=make_enricher(
            {"type": "dummy"},
            field="category",
            vocabulary=["Living Room"],
        ),
    )

    matches = strategy._category_matches(["Living Room"])

    assert matches.tolist() == [True, False, False, False]


def test_query_understanding_limits_category_vocabulary():
    categories = [f"category-{index}" for index in range(MAX_CATEGORY_CARDINALITY + 1)]
    categories.append(categories[-1])
    corpus = pd.DataFrame(
        {
            "title": ["title"] * len(categories),
            "description": ["description"] * len(categories),
            "category": categories,
        }
    )

    with pytest.warns(UserWarning, match="top 300"):
        strategy = QueryUnderstandingStrategy.build(
            {
                "categorize": {
                    "field": "category",
                    "enrichment_engine": {"type": "dummy"},
                },
                "retrieval_engine": {
                    "base": "bm25_boosted",
                    "params": {"fields": ["title^1"]},
                },
            },
            corpus=corpus,
        )

    vocabulary = strategy.enricher.vocabulary
    assert len(vocabulary) == MAX_CATEGORY_CARDINALITY
    assert categories[-1] in vocabulary
    assert len(set(categories[:-1]) - set(vocabulary)) == 1
