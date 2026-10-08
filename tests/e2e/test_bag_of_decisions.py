from types import SimpleNamespace
from unittest.mock import patch

from exps.runners.run import RunParams, run_benchmark


class ScriptedQuestionGenerator:
    instances = []

    def __init__(self, **kwargs):
        self.response_model = kwargs["response_model"]
        self.cached_calls = []
        self.uncached_calls = []
        self.enricher = SimpleNamespace(enrich=self.enrich_uncached)
        self.__class__.instances.append(self)

    def enrich_uncached(self, prompt):
        self.uncached_calls.append(prompt)
        return self.response_model(decisions=["Does this document match the query?"])

    def enrich(self, prompt):
        self.cached_calls.append(prompt)
        return self.response_model(decisions=["Does this document match the query?"])


class ScriptedDecisionClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        probability = 0.1 if len(self.calls) == 1 else 0.95
        answers = {
            question_id: type("Answer", (), {"noul": probability})()
            for question_id in questions
        }
        return type("Response", (), {"answers": answers})()


def test_run_benchmark_bag_of_decisions_reranks_candidate_documents(
    tmp_path, doug_blog_dataset
):
    ScriptedQuestionGenerator.instances = []
    ScriptedDecisionClient.instances = []
    query = doug_blog_dataset.judgments.iloc[0]["query"]
    config_path = tmp_path / "bag_of_decisions.yml"
    config_path.write_text(
        """
strategy:
  name: bag_of_decisions_fixture
  type: bag_of_decisions
  params:
    decision_engine:
      generator:
        type: llm
        system_prompt: Generate relevance questions.
        prompt: Generate yes/no questions for {query}.
        model: gpt-5
      reranker:
        decision_model: jev/jev-latest
        decision_weight: 1000
        confidence_threshold: 0.7
        k: 2
        state_format: |
          {doc_id}
          {title}
          {description}
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title]
""".lstrip(),
        encoding="utf-8",
    )

    with (
        patch(
            "exps.bag_of_decisions.decision_generator.AutoEnricher",
            ScriptedQuestionGenerator,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.TypeSafeClient",
            ScriptedDecisionClient,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.key_for_provider",
            return_value="typesafe-test-key",
        ),
    ):
        result = run_benchmark(
            RunParams(
                strategy_path=str(config_path),
                dataset="doug_blog",
                query=query,
                k=2,
                no_cache=True,
            )
        )

    assert result.strategy_name == "bag_of_decisions_fixture"
    assert result.query_results is not None
    client = ScriptedDecisionClient.instances[0]
    assert len(client.calls) == 2
    candidate_states = [state for state, _ in client.calls]
    # The second of the two L0 candidates gets the stronger Noul score and
    # should move above the first candidate after reranking.
    assert result.query_results.iloc[0]["doc_id"] == int(
        candidate_states[1].splitlines()[0]
    )
    assert ScriptedQuestionGenerator.instances[0].cached_calls == []
    assert ScriptedQuestionGenerator.instances[0].uncached_calls == [
        f"Generate yes/no questions for {query}."
    ]


def test_run_benchmark_direct_generator_skips_llm(tmp_path, doug_blog_dataset):
    ScriptedDecisionClient.instances = []
    query = doug_blog_dataset.judgments.iloc[0]["query"]
    config_path = tmp_path / "bag_of_decisions_direct.yml"
    config_path.write_text(
        """
strategy:
  name: bag_of_decisions_direct_fixture
  type: bag_of_decisions
  params:
    decision_engine:
      generator:
        type: direct
        question: |
          Is this document relevant to the search query: {query}?
      reranker:
        decision_model: jev/jev-latest
        decision_weight: 1000
        confidence_threshold: 0.7
        k: 2
        state_format: |
          {doc_id}
          {title}
          {description}
    retrieval_engine:
      base: bm25_boosted
      params:
        fields: [title]
""".lstrip(),
        encoding="utf-8",
    )

    with (
        patch(
            "exps.bag_of_decisions.decision_generator.AutoEnricher",
            side_effect=AssertionError("direct generator must not call AutoEnricher"),
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.TypeSafeClient",
            ScriptedDecisionClient,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.key_for_provider",
            return_value="typesafe-test-key",
        ),
    ):
        result = run_benchmark(
            RunParams(
                strategy_path=str(config_path),
                dataset="doug_blog",
                query=query,
                k=2,
                no_cache=True,
            )
        )

    assert result.strategy_name == "bag_of_decisions_direct_fixture"
    client = ScriptedDecisionClient.instances[0]
    assert len(client.calls) == 2
    expected_question = f"Is this document relevant to the search query: {query}?"
    for _, questions in client.calls:
        assert len(questions) == 1
        assert next(iter(questions.values())).instructions == expected_question
