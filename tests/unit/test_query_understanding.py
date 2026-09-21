from unittest.mock import patch

from exps.query_understanding.enrichers import make_llm_single_enricher


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
    assert '"category"' in auto_enricher.prompts[0]
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
            )
            assert enricher.enrich("ambiguous") == []
    finally:
        FakeAutoEnricher.value = "Furniture"
