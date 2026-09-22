from unittest.mock import patch

import pandas as pd
import pytest
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer

from exps.query_understanding.enrichers import make_enricher
from exps.query_understanding.enrichers import make_llm_multiple_enricher
from exps.query_understanding.enrichers import make_llm_single_enricher
from exps.query_understanding.enrichers import make_choice_single_enricher
from exps.query_understanding import QueryUnderstandingStrategy
from exps.query_understanding.retrieval_engines import (
    BM25HierarchyBoostedRetrievalEngine,
)
from exps.query_understanding.enrichers.llm_multiple import _model_name as multiple_model_name
from exps.query_understanding.enrichers.llm_single import _model_name as single_model_name
from exps.query_understanding.strategy import MAX_CATEGORY_CARDINALITY
from exps.runners.query_classification import _evaluate_as, _ground_truth


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


class FakeJevClient:
    instances = []
    value = "Furniture"
    confidence = 1.0

    def __init__(self, **kwargs):
        self.api_key = kwargs["api_key"]
        self.model = kwargs["model"]
        self.calls = []
        self.__class__.instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        question_id = next(iter(questions))
        answer = type(
            "Answer", (), {"choice": self.value, "confidence": self.confidence}
        )()
        return type("Response", (), {"choices": {question_id: answer}})()


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


def test_llm_enricher_model_names_default_to_openai_without_overwriting_provider():
    assert single_model_name("gpt-5") == "openai/gpt-5"
    assert single_model_name("openai/gpt-5") == "openai/gpt-5"
    assert multiple_model_name("gpt-5-mini") == "openai/gpt-5-mini"
    assert multiple_model_name("anthropic/claude-sonnet") == "anthropic/claude-sonnet"


def test_llm_multiple_enricher_returns_deduplicated_categories():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = ["Furniture", "Lighting", "Unknown", "Furniture"]
    try:
        with patch(
            "exps.query_understanding.enrichers.llm_multiple.AutoEnricher",
            FakeAutoEnricher,
        ):
            enricher = make_llm_multiple_enricher(
                field="category hierarchy",
                vocabulary=["Furniture", "Lighting", "Outdoor"],
                prompt="Classify {query} into {field} values.",
            )
            assert enricher.enrich("sofa") == ["Furniture", "Lighting"]
            assert enricher.enrich("sofa") == ["Furniture", "Lighting"]

        auto_enricher = FakeAutoEnricher.instances[0]
        assert auto_enricher.prompts == ["Classify sofa into category hierarchy values."]
    finally:
        FakeAutoEnricher.value = "Furniture"


def test_llm_multiple_enricher_requires_prompt():
    with pytest.raises(ValueError, match="params.prompt"):
        make_enricher(
            {"type": "llm_multiple"},
            field="category",
            vocabulary=["Furniture"],
        )


def test_llm_multiple_enricher_supports_quoted_category_values():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = ['__category_0__', 'Furniture']
    try:
        with patch(
            "exps.query_understanding.enrichers.llm_multiple.AutoEnricher",
            FakeAutoEnricher,
        ):
            enricher = make_llm_multiple_enricher(
                field="category hierarchy",
                vocabulary=['Bar (28"-33") Stools', "Furniture"],
                prompt="Classify {query} into {field} values.",
            )
            assert enricher.enrich("bar stool") == ['Bar (28"-33") Stools', "Furniture"]
            assert "__category_0__" in FakeAutoEnricher.instances[0].prompts[0]
    finally:
        FakeAutoEnricher.value = "Furniture"


def test_choice_single_enricher_adds_choice_descriptions_to_prompt():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = "Furniture"
    with patch(
        "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_choice_single_enricher(
            field="category",
            vocabulary=["Furniture", "Lighting"],
            choices="Furniture: Products used to furnish a room.\nLighting: Products that provide illumination.",
            prompt="Which best describes the query?\n{query}",
            model="gpt-5-mini",
        )
        assert enricher.enrich("sofa") == ["Furniture"]

    prompt = FakeAutoEnricher.instances[0].prompts[0]
    assert "Which best describes the query?\nsofa" in prompt
    assert "- Furniture: Products used to furnish a room." in prompt
    assert "- Lighting: Products that provide illumination." in prompt


def test_choice_single_enricher_can_pad_missing_vocabulary_choices():
    FakeAutoEnricher.instances = []
    with patch(
        "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_choice_single_enricher(
            field="category",
            vocabulary=["Furniture", "Lighting"],
            choices={"Furniture": "Products used to furnish a room."},
            prompt="Classify {query}.",
            params={"pad_missing_choices": True},
        )
    assert "Lighting" in enricher.response_model.model_json_schema()["properties"]["choice"]["enum"]


def test_choice_single_unknown_means_no_classification():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = "Unknown"
    try:
        with patch(
            "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
            FakeAutoEnricher,
        ):
            enricher = make_choice_single_enricher(
                field="category",
                vocabulary=["Furniture"],
                choices={"Furniture": "Products used to furnish a room."},
                prompt="Classify {query}.",
            )
            assert enricher.enrich("ambiguous") == []
            assert "- Unknown: No classification applies." in FakeAutoEnricher.instances[0].prompts[0]
    finally:
        FakeAutoEnricher.value = "Furniture"


def test_choice_single_jev_uses_structured_criteria_and_normalizes_unknown():
    FakeJevClient.instances = []
    FakeJevClient.value = "Unknown"
    try:
        with patch(
            "exps.query_understanding.enrichers.choice_single_jev.TypeSafeClient",
            FakeJevClient,
        ), patch(
            "exps.query_understanding.enrichers.choice_single_jev.key_for_provider",
            return_value="typesafe-test-key",
        ):
            enricher = make_enricher(
                {
                    "type": "choice_single",
                    "params": {
                        "model": "jev/jev-latest",
                        "choices": {
                            "Furniture": "Products used to furnish a room.",
                        },
                        "prompt": "Which category fits {query}?",
                    },
                },
                field="category",
                vocabulary=["Furniture", "Lighting"],
            )
            assert enricher.enrich("ambiguous") == []

        client = FakeJevClient.instances[0]
        assert client.api_key == "typesafe-test-key"
        assert client.model == "jev-latest"
        state, questions = client.calls[0]
        assert state == "ambiguous"
        question = questions["category"]
        assert question.instructions == "Which category fits ambiguous?"
        assert question.criteria == {
            "Furniture": "Products used to furnish a room.",
            "Unknown": "No classification applies.",
        }
    finally:
        FakeJevClient.value = "Furniture"


def test_choice_single_jev_requires_confidence_to_be_strictly_above_threshold():
    FakeJevClient.instances = []
    FakeJevClient.value = "Furniture"
    FakeJevClient.confidence = 0.5
    try:
        with patch(
            "exps.query_understanding.enrichers.choice_single_jev.TypeSafeClient",
            FakeJevClient,
        ), patch(
            "exps.query_understanding.enrichers.choice_single_jev.key_for_provider",
            return_value="typesafe-test-key",
        ):
            enricher = make_enricher(
                {
                    "type": "choice_single",
                    "params": {
                        "model": "jev/jev-latest",
                        "confidence_threshold": 0.5,
                        "choices": {"Furniture": "Products used to furnish a room."},
                        "prompt": "Classify {query}.",
                    },
                },
                field="category",
                vocabulary=["Furniture"],
            )
            assert enricher.enrich("borderline") == []

        FakeJevClient.confidence = 0.51
        with patch(
            "exps.query_understanding.enrichers.choice_single_jev.TypeSafeClient",
            FakeJevClient,
        ), patch(
            "exps.query_understanding.enrichers.choice_single_jev.key_for_provider",
            return_value="typesafe-test-key",
        ):
            enricher = make_enricher(
                {
                    "type": "choice_single",
                    "params": {
                        "model": "jev/jev-latest",
                        "confidence_threshold": 0.5,
                        "choices": {"Furniture": "Products used to furnish a room."},
                        "prompt": "Classify {query}.",
                    },
                },
                field="category",
                vocabulary=["Furniture"],
            )
            assert enricher.enrich("above-threshold") == ["Furniture"]
    finally:
        FakeJevClient.value = "Furniture"
        FakeJevClient.confidence = 1.0


def test_choice_single_rejects_confidence_threshold_for_openai():
    with pytest.raises(ValueError, match="only supported for Jev"):
        make_choice_single_enricher(
            field="category",
            vocabulary=["Furniture"],
            choices={"Furniture": "Products used to furnish a room."},
            prompt="Classify {query}.",
            params={"confidence_threshold": 0.5},
        )


def test_choice_single_unprefixed_model_defaults_to_openai():
    FakeAutoEnricher.instances = []
    with patch(
        "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_choice_single_enricher(
            field="category",
            vocabulary=["Furniture"],
            choices={"Furniture": "Products used to furnish a room."},
            prompt="Classify {query}.",
            model="jev-latest",
        )
    assert enricher.model == "openai/jev-latest"


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


def test_classification_metrics_only_average_queries_with_predictions():
    judgments = pd.DataFrame(
        {
            "query": ["sofa", "lamp"],
            "doc_id": [1, 2],
            "grade": [1, 1],
        }
    )
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2],
            "category": ["Furniture", "Lighting"],
        }
    )

    result = _evaluate_as(
        eval_as="direct",
        queries=["sofa", "lamp"],
        judgments=judgments,
        corpus=corpus,
        category_field="category",
        threshold=0.8,
        predictions={"sofa": ["Furniture"], "lamp": []},
    )

    assert result.mean_recall == 1.0
    assert result.mean_jaccard == 1.0
    assert result.coverage == 0.5


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


def test_hierarchy_boosting_sums_decaying_prefix_boosts_per_classification():
    corpus = pd.DataFrame(
        {
            "category": [
                "foo / bar / baz",
                "foo / bar / qux",
                "foo / other",
                "other",
            ]
        }
    )
    corpus["category_snowball"] = SearchArray.index(
        corpus["category"], snowball_tokenizer
    )
    engine = BM25HierarchyBoostedRetrievalEngine(
        corpus,
        "category_snowball",
        {"boost_matches": 10, "decay": 0.5},
    )

    scores = engine.apply(
        pd.Series([0.0] * len(corpus)).to_numpy(),
        ["foo / bar / baz", "other"],
    )

    assert scores.tolist() == [17.5, 15.0, 20.0, 10.0]


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
