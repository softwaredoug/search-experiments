from unittest.mock import patch

import pandas as pd
import pytest
from searcharray import SearchArray
from cheat_at_search.tokenizers import snowball_tokenizer

from exps.query_understanding.enrichers import make_enricher
from exps.query_understanding.enrichers import make_llm_multiple_enricher
from exps.query_understanding.enrichers import make_llm_single_enricher
from exps.query_understanding.enrichers import (
    make_llm_choice_enricher,
)
from exps.query_understanding.enrichers.cached_choice_single_jev import (
    CachedJevChoiceSingleEnricher,
)
from exps.query_understanding.enrichers.cached_jev_choice_multiple import (
    CachedJevChoiceMultipleEnricher,
)
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


class FakeJevMultipleClient:
    instances = []
    distributions = {}

    def __init__(self, **kwargs):
        self.api_key = kwargs["api_key"]
        self.model = kwargs["model"]
        self.calls = []
        self.__class__.instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        question_id = next(iter(questions))
        probabilities = self.distributions[state]
        answer = type(
            "Answer",
            (),
            {
                "choice": max(probabilities, key=probabilities.get)
                if probabilities
                else None,
                "confidence": max(probabilities.values()) if probabilities else 0.0,
                "probabilities": probabilities,
            },
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


def test_llm_choice_enricher_adds_choice_descriptions_to_prompt():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = "Furniture"
    with patch(
        "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_llm_choice_enricher(
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


def test_llm_choice_enricher_can_pad_missing_vocabulary_choices():
    FakeAutoEnricher.instances = []
    with patch(
        "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_llm_choice_enricher(
            field="category",
            vocabulary=["Furniture", "Lighting"],
            choices={"Furniture": "Products used to furnish a room."},
            prompt="Classify {query}.",
            params={"pad_missing_choices": True},
        )
    assert "Lighting" in enricher.response_model.model_json_schema()["properties"]["choice"]["enum"]


def test_llm_choice_enricher_uses_unlabeled_vocabulary_when_choices_are_omitted():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = "Furniture"
    with patch(
        "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_enricher(
            {
                "type": "llm_choice",
                "params": {"prompt": "Classify {query}."},
            },
            field="category",
            vocabulary=["Furniture", "Lighting"],
        )
        assert enricher.enrich("sofa") == ["Furniture"]

    assert "Furniture" in enricher.response_model.model_json_schema()["properties"]["choice"]["enum"]
    assert "Lighting" in enricher.response_model.model_json_schema()["properties"]["choice"]["enum"]
    assert "- Furniture\n- Lighting" in FakeAutoEnricher.instances[0].prompts[0]


def test_llm_choice_unknown_means_no_classification():
    FakeAutoEnricher.instances = []
    FakeAutoEnricher.value = "Unknown"
    try:
        with patch(
            "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
            FakeAutoEnricher,
        ):
            enricher = make_llm_choice_enricher(
                field="category",
                vocabulary=["Furniture"],
                choices={"Furniture": "Products used to furnish a room."},
                prompt="Classify {query}.",
            )
            assert enricher.enrich("ambiguous") == []
            assert "- Unknown: No classification applies." in FakeAutoEnricher.instances[0].prompts[0]
    finally:
        FakeAutoEnricher.value = "Furniture"


def test_jev_choice_single_uses_structured_criteria_and_normalizes_unknown(tmp_path):
    FakeJevClient.instances = []
    FakeJevClient.value = "Unknown"
    try:
        with patch(
            "exps.query_understanding.enrichers.choice_single_jev.TypeSafeClient",
            FakeJevClient,
        ), patch(
            "exps.query_understanding.enrichers.choice_single_jev.key_for_provider",
            return_value="typesafe-test-key",
        ), patch(
            "exps.query_understanding.enrichers.cached_choice_single_jev.DATA_PATH",
            tmp_path,
        ):
            enricher = make_enricher(
                {
                    "type": "jev_choice_single",
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
            assert isinstance(enricher, CachedJevChoiceSingleEnricher)
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


def test_jev_choice_single_uses_popularity_ordered_unlabeled_choices_when_empty(
    tmp_path,
):
    FakeJevClient.instances = []
    FakeJevClient.value = "category-0"
    vocabulary = [f"category-{index}" for index in range(260)]
    try:
        with patch(
            "exps.query_understanding.enrichers.choice_single_jev.TypeSafeClient",
            FakeJevClient,
        ), patch(
            "exps.query_understanding.enrichers.choice_single_jev.key_for_provider",
            return_value="typesafe-test-key",
        ), patch(
            "exps.query_understanding.enrichers.cached_choice_single_jev.DATA_PATH",
            tmp_path,
        ):
            enricher = make_enricher(
                {
                    "type": "jev_choice_single",
                    "params": {
                        "model": "jev/jev-latest",
                        "choices": {},
                        "prompt": "Classify {query}.",
                    },
                },
                field="category",
                vocabulary=vocabulary,
            )
            assert enricher.enrich("query") == ["category-0"]

        criteria = FakeJevClient.instances[0].calls[0][1]["category"].criteria
        assert len(criteria) == 255
        assert list(criteria) == [*vocabulary[:254], "Unknown"]
        assert all(criteria[value] is None for value in vocabulary[:254])
        assert criteria["Unknown"] == "No classification applies."
    finally:
        FakeJevClient.value = "Furniture"


def test_jev_choice_single_requires_confidence_to_be_strictly_above_threshold(
    tmp_path,
):
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
        ), patch(
            "exps.query_understanding.enrichers.cached_choice_single_jev.DATA_PATH",
            tmp_path,
        ):
            enricher = make_enricher(
                {
                    "type": "jev_choice_single",
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
        ), patch(
            "exps.query_understanding.enrichers.cached_choice_single_jev.DATA_PATH",
            tmp_path,
        ):
            enricher = make_enricher(
                {
                    "type": "jev_choice_single",
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


def test_jev_choice_multiple_thresholds_probabilities_and_caches_predictions(
    tmp_path,
):
    FakeJevMultipleClient.instances = []
    FakeJevMultipleClient.distributions = {
        "bed": {"Furniture": 0.8, "Bedroom": 0.4, "Outdoor": 0.5},
        "unmatched": {},
    }
    with patch(
        "exps.query_understanding.enrichers.jev_choice_multiple.TypeSafeClient",
        FakeJevMultipleClient,
    ), patch(
        "exps.query_understanding.enrichers.jev_choice_multiple.key_for_provider",
        return_value="typesafe-test-key",
    ), patch(
        "exps.query_understanding.enrichers.cached_jev_choice_multiple.DATA_PATH",
        tmp_path,
    ):
        enricher = make_enricher(
            {
                "type": "jev_choice_multiple",
                "params": {
                    "model": "jev/jev-latest",
                    "threshold": 0.4,
                    "prompt": "Classify {query}.",
                },
            },
            field="category",
            vocabulary=["Furniture", "Bedroom", "Outdoor"],
        )
        assert isinstance(enricher, CachedJevChoiceMultipleEnricher)
        assert enricher.enrich("bed") == ["Furniture", "Outdoor"]
        assert enricher.enrich("bed") == ["Furniture", "Outdoor"]
        assert enricher.enrich("unmatched") == []

    client = FakeJevMultipleClient.instances[0]
    assert [state for state, _ in client.calls] == ["bed", "unmatched"]
    assert client.calls[0][1]["category"].criteria == {
        "Furniture": None,
        "Bedroom": None,
        "Outdoor": None,
    }


def test_jev_choice_multiple_uses_up_to_255_most_frequent_options(tmp_path):
    FakeJevMultipleClient.instances = []
    vocabulary = [f"category-{index}" for index in range(260)]
    FakeJevMultipleClient.distributions = {
        "query": {
            "category-0": 0.41,
            "category-254": 0.5,
            "category-255": 0.99,
            "Unknown": 1.0,
        }
    }
    with patch(
        "exps.query_understanding.enrichers.jev_choice_multiple.TypeSafeClient",
        FakeJevMultipleClient,
    ), patch(
        "exps.query_understanding.enrichers.jev_choice_multiple.key_for_provider",
        return_value="typesafe-test-key",
    ), patch(
        "exps.query_understanding.enrichers.cached_jev_choice_multiple.DATA_PATH",
        tmp_path,
    ):
        enricher = make_enricher(
            {
                "type": "jev_choice_multiple",
                "params": {
                    "model": "jev/jev-latest",
                    "threshold": 0.4,
                    "prompt": "Classify {query}.",
                },
            },
            field="category",
            vocabulary=vocabulary,
        )
        assert enricher.enrich("query") == ["category-0", "category-254"]

    criteria = FakeJevMultipleClient.instances[0].calls[0][1]["category"].criteria
    assert len(criteria) == 255
    assert list(criteria) == vocabulary[:255]
    assert "Unknown" not in criteria


def test_jev_choice_multiple_accepts_descriptions_and_validates_threshold_and_model(
    tmp_path,
):
    FakeJevMultipleClient.instances = []
    FakeJevMultipleClient.distributions = {
        "query": {"Furniture": 0.41, "Bedroom": 0.7, "Unknown": 0.9}
    }
    with patch(
        "exps.query_understanding.enrichers.jev_choice_multiple.TypeSafeClient",
        FakeJevMultipleClient,
    ), patch(
        "exps.query_understanding.enrichers.jev_choice_multiple.key_for_provider",
        return_value="typesafe-test-key",
    ), patch(
        "exps.query_understanding.enrichers.cached_jev_choice_multiple.DATA_PATH",
        tmp_path,
    ):
        enricher = make_enricher(
            {
                "type": "jev_choice_multiple",
                "params": {
                    "model": "jev/jev-latest",
                    "threshold": 0.4,
                    "choices": {
                        "Furniture": "Products used to furnish a room.",
                        "Bedroom": None,
                        "Unknown": "No category applies.",
                    },
                    "prompt": "Classify {query}.",
                },
            },
            field="category",
            vocabulary=["Furniture", "Bedroom", "Outdoor"],
        )
        assert enricher.enrich("query") == ["Bedroom", "Furniture"]

    criteria = FakeJevMultipleClient.instances[0].calls[0][1]["category"].criteria
    assert criteria == {
        "Furniture": "Products used to furnish a room.",
        "Bedroom": None,
        "Unknown": "No category applies.",
    }

    with pytest.raises(ValueError, match="threshold"):
        make_enricher(
            {
                "type": "jev_choice_multiple",
                "params": {"model": "jev/jev-latest", "prompt": "Classify {query}."},
            },
            field="category",
            vocabulary=["Furniture"],
        )
    with pytest.raises(ValueError, match="requires a Jev model"):
        make_enricher(
            {
                "type": "jev_choice_multiple",
                "params": {
                    "model": "gpt-5-mini",
                    "threshold": 0.4,
                    "prompt": "Classify {query}.",
                },
            },
            field="category",
            vocabulary=["Furniture"],
        )


def test_cached_jev_choice_single_persists_predictions_and_empty_results(tmp_path):
    from exps.query_understanding.enrichers import cached_choice_single_jev

    class FakeJevEnricher:
        instances = []

        def __init__(self, **kwargs):
            self.cache_key = "fixture-cache-key"
            self.calls = []
            self.__class__.instances.append(self)

        def enrich(self, query):
            self.calls.append(query)
            return {"chair": ["Furniture"], "ambiguous": []}[query]

    FakeJevEnricher.instances = []
    with patch.object(cached_choice_single_jev, "DATA_PATH", tmp_path), patch.object(
        cached_choice_single_jev, "JevChoiceSingleEnricher", FakeJevEnricher
    ):
        kwargs = {
            "field": "category",
            "vocabulary": ["Furniture"],
            "choices": {"Furniture": "Products used to furnish a room."},
            "prompt": "Classify {query}.",
            "model": "jev/jev-latest",
            "reasoning": None,
            "pad_missing_choices": False,
        }
        first = cached_choice_single_jev.CachedJevChoiceSingleEnricher(**kwargs)
        assert first.enrich("chair") == ["Furniture"]
        assert first.enrich("ambiguous") == []

        second = cached_choice_single_jev.CachedJevChoiceSingleEnricher(**kwargs)
        assert second.enrich("chair") == ["Furniture"]
        assert second.enrich("ambiguous") == []

    assert FakeJevEnricher.instances[0].calls == ["chair", "ambiguous"]
    assert FakeJevEnricher.instances[1].calls == []
    cache_path = tmp_path / "query_understanding_cache" / "fixture-cache-key.json"
    assert cache_path.read_text(encoding="utf-8") == (
        '{"ambiguous": [], "chair": ["Furniture"]}\n'
    )


def test_cached_jev_choice_multiple_persists_multiple_predictions(tmp_path):
    from exps.query_understanding.enrichers import cached_jev_choice_multiple

    class FakeJevMultipleEnricher:
        instances = []

        def __init__(self, **kwargs):
            self.cache_key = "fixture-multiple-cache-key"
            self.calls = []
            self.__class__.instances.append(self)

        def enrich(self, query):
            self.calls.append(query)
            return {"bed": ["Bedroom", "Furniture"]}[query]

    FakeJevMultipleEnricher.instances = []
    with patch.object(cached_jev_choice_multiple, "DATA_PATH", tmp_path), patch.object(
        cached_jev_choice_multiple,
        "JevChoiceMultipleEnricher",
        FakeJevMultipleEnricher,
    ):
        kwargs = {
            "field": "category",
            "vocabulary": ["Furniture", "Bedroom"],
            "choices": {"Furniture": None, "Bedroom": None},
            "prompt": "Classify {query}.",
            "model": "jev/jev-latest",
            "threshold": 0.4,
            "reasoning": None,
            "pad_missing_choices": False,
        }
        first = cached_jev_choice_multiple.CachedJevChoiceMultipleEnricher(**kwargs)
        assert first.enrich("bed") == ["Bedroom", "Furniture"]

        second = cached_jev_choice_multiple.CachedJevChoiceMultipleEnricher(**kwargs)
        assert second.enrich("bed") == ["Bedroom", "Furniture"]

    assert FakeJevMultipleEnricher.instances[0].calls == ["bed"]
    assert FakeJevMultipleEnricher.instances[1].calls == []
    cache_path = tmp_path / "query_understanding_cache" / "fixture-multiple-cache-key.json"
    assert cache_path.read_text(encoding="utf-8") == (
        '{"bed": ["Bedroom", "Furniture"]}\n'
    )


def test_llm_choice_rejects_confidence_threshold():
    with pytest.raises(ValueError, match="only supported for Jev"):
        make_llm_choice_enricher(
            field="category",
            vocabulary=["Furniture"],
            choices={"Furniture": "Products used to furnish a room."},
            prompt="Classify {query}.",
            params={"confidence_threshold": 0.5},
        )


def test_choice_engines_reject_the_other_provider():
    for model in ("jev/jev-latest", "google/gemini-2.5-flash"):
        with pytest.raises(ValueError, match="OpenAI models only"):
            make_enricher(
                {
                    "type": "llm_choice",
                    "params": {
                        "model": model,
                        "prompt": "Classify {query}.",
                    },
                },
                field="category",
                vocabulary=["Furniture"],
            )

    with pytest.raises(ValueError, match="requires a Jev model"):
        make_enricher(
            {
                "type": "jev_choice_single",
                "params": {
                    "model": "gpt-5-mini",
                    "prompt": "Classify {query}.",
                },
            },
            field="category",
            vocabulary=["Furniture"],
        )


def test_choice_single_engine_type_is_no_longer_supported():
    with pytest.raises(ValueError, match="Supported enrichment engines"):
        make_enricher(
            {"type": "choice_single"},
            field="category",
            vocabulary=["Furniture"],
        )


def test_llm_choice_unprefixed_model_defaults_to_openai():
    FakeAutoEnricher.instances = []
    with patch(
        "exps.query_understanding.enrichers.choice_single_openai.AutoEnricher",
        FakeAutoEnricher,
    ):
        enricher = make_llm_choice_enricher(
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


def test_classification_metrics_score_empty_category_sets():
    judgments = pd.DataFrame(
        {
            "query": ["both-empty", "both-empty", "false-positive", "false-positive", "miss"],
            "doc_id": [1, 2, 1, 2, 3],
            "grade": [1, 1, 1, 1, 1],
        }
    )
    corpus = pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "category": ["Furniture", "Lighting", "Garden"],
        }
    )

    result = _evaluate_as(
        eval_as="direct",
        queries=["both-empty", "false-positive", "miss"],
        judgments=judgments,
        corpus=corpus,
        category_field="category",
        threshold=0.8,
        predictions={
            "both-empty": [],
            "false-positive": ["Outdoor"],
            "miss": [],
        },
    )

    assert result.per_query["expected_categories"].tolist() == [[], [], ["Garden"]]
    assert result.per_query["recall"].tolist() == [1.0, 0.0, 0.0]
    assert result.per_query["jaccard"].tolist() == [1.0, 0.0, 0.0]
    assert result.mean_recall == 0.0
    assert result.mean_jaccard == 0.0
    assert result.coverage == 1 / 3


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
    assert vocabulary[0] == categories[-1]
    assert categories[-1] in vocabulary
    assert len(set(categories[:-1]) - set(vocabulary)) == 1
