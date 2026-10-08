from unittest.mock import patch

import pandas as pd
import pytest

from exps.bag_of_decisions import BagOfDecisionsStrategy
from exps.bag_of_decisions.direct_decision_generator import DirectDecisionGenerator


class ScriptedAutoEnricher:
    decisions = ["Does this document match the query?"]
    instances = []

    def __init__(self, **kwargs):
        self.response_model = kwargs["response_model"]
        self.kwargs = kwargs
        self.prompts = []
        self.__class__.instances.append(self)

    def enrich(self, prompt):
        self.prompts.append(prompt)
        return self.response_model(decisions=self.decisions)


class ScriptedNoulClient:
    probabilities = {}
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        answers = {
            question_id: type(
                "Answer", (), {"noul": self.probabilities[state][question_id]}
            )()
            for question_id in questions
        }
        return type("Response", (), {"answers": answers})()


def _params(*, generator_overrides=None, reranker_overrides=None):
    generator = {
        "type": "llm",
        "system_prompt": "Generate search relevance decisions.",
        "prompt": "Generate yes/no questions for {query}.",
        "model": "gpt-5",
    }
    generator.update(generator_overrides or {})
    reranker = {
        "decision_model": "jev/jev-latest",
        "decision_weight": 10,
        "k": 2,
        "state_format": "{query}\n{title}\n{description}",
    }
    reranker.update(reranker_overrides or {})
    return {
        "decision_engine": {"generator": generator, "reranker": reranker},
        "retrieval_engine": {
            "base": "bm25_boosted",
            "params": {"fields": ["title"]},
        },
    }


def _corpus():
    return pd.DataFrame(
        {
            "doc_id": [1, 2, 3],
            "title": ["desk alpha", "desk beta", "desk gamma"],
            "description": ["first", "second", "third"],
        }
    )


def test_direct_decision_generator_formats_a_single_query_question():
    generator = DirectDecisionGenerator(
        question='Is this document relevant to "{query}"?',
        criteria={
            "true": "The product matches the query's product and shopping intent.",
            "false": "The product is unrelated or does not satisfy the query.",
        },
    )

    decision = generator.generate("desk lamp")[0]
    assert decision.instructions == 'Is this document relevant to "desk lamp"?'
    assert decision.criteria == {
        "true": "The product matches the query's product and shopping intent.",
        "false": "The product is unrelated or does not satisfy the query.",
    }


@pytest.mark.parametrize("question", ["", "   ", "Does this fit?", "Does this fit {other}?"])
def test_direct_decision_generator_requires_a_query_template(question):
    with pytest.raises(ValueError, match="question.*query"):
        DirectDecisionGenerator(question=question)


def test_direct_decision_generator_validates_criteria():
    with pytest.raises(ValueError, match="criteria.*true.*false"):
        DirectDecisionGenerator(
            question="Is this relevant to {query}?",
            criteria={"true": "Relevant"},
        )


def test_build_generates_questions_and_scores_all_yes_probabilities():
    ScriptedAutoEnricher.instances = []
    ScriptedAutoEnricher.decisions = ["Is this a desk?", "Is this suitable for work?"]
    ScriptedNoulClient.instances = []
    ScriptedNoulClient.probabilities = {
        "desk\ndesk alpha\nfirst": {"decision_0": 0.8, "decision_1": 0.7},
        "desk\ndesk beta\nsecond": {"decision_0": 0.9, "decision_1": 0.95},
    }

    with (
        patch(
            "exps.bag_of_decisions.decision_generator.AutoEnricher",
            ScriptedAutoEnricher,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.TypeSafeClient",
            ScriptedNoulClient,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.key_for_provider",
            return_value="typesafe-test-key",
        ),
    ):
        strategy = BagOfDecisionsStrategy.build(_params(), corpus=_corpus())
        indices, scores = strategy.search("desk", k=3)

    client = ScriptedNoulClient.instances[0]
    assert [state for state, _ in client.calls] == [
        "desk\ndesk alpha\nfirst",
        "desk\ndesk beta\nsecond",
    ]
    questions = client.calls[0][1]
    assert list(questions) == ["decision_0", "decision_1"]
    assert [question.instructions for question in questions.values()] == [
        "Is this a desk?",
        "Is this suitable for work?",
    ]
    assert all(question.criteria is None for question in questions.values())
    assert ScriptedAutoEnricher.instances[0].prompts == [
        "Generate yes/no questions for desk."
    ]
    assert indices.tolist() == [1, 0, 2]
    assert scores[0] - scores[1] == pytest.approx(3.5)


def test_noul_score_sums_all_yes_probabilities_without_thresholding():
    ScriptedAutoEnricher.instances = []
    ScriptedAutoEnricher.decisions = ["first?", "second?", "third?"]
    ScriptedNoulClient.instances = []
    ScriptedNoulClient.probabilities = {
        "desk\ndesk alpha\nfirst": {
            "decision_0": 0.7,
            "decision_1": 0.30001,
            "decision_2": 0.99,
        },
        "desk\ndesk beta\nsecond": {
            "decision_0": 0.1,
            "decision_1": 0.2,
            "decision_2": 0.3,
        },
    }

    with (
        patch(
            "exps.bag_of_decisions.decision_generator.AutoEnricher",
            ScriptedAutoEnricher,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.TypeSafeClient",
            ScriptedNoulClient,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.key_for_provider",
            return_value="typesafe-test-key",
        ),
    ):
        strategy = BagOfDecisionsStrategy.build(_params(), corpus=_corpus())
        indices, scores = strategy.search("desk", k=3)

    assert indices.tolist()[0] == 0
    # All probabilities contribute, including low P(yes) and uncertain values.
    score_sum_a = 0.7 + 0.30001 + 0.99
    score_sum_b = 0.1 + 0.2 + 0.3
    assert scores[0] - scores[1] == pytest.approx((score_sum_a - score_sum_b) * 10)


def test_empty_generated_decisions_leave_retrieval_scores_unchanged():
    ScriptedAutoEnricher.instances = []
    ScriptedAutoEnricher.decisions = []
    ScriptedNoulClient.instances = []

    with (
        patch(
            "exps.bag_of_decisions.decision_generator.AutoEnricher",
            ScriptedAutoEnricher,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.TypeSafeClient",
            ScriptedNoulClient,
        ),
        patch(
            "exps.bag_of_decisions.decision_reranker.key_for_provider",
            return_value="typesafe-test-key",
        ),
    ):
        strategy = BagOfDecisionsStrategy.build(_params(), corpus=_corpus())
        baseline = strategy._baseline_scores("desk")
        indices, scores = strategy.search("desk", k=3)

    assert ScriptedNoulClient.instances[0].calls == []
    assert scores.tolist() == baseline[indices].tolist()


@pytest.mark.parametrize(
    ("section", "updates", "error"),
    [
        ("generator", {"prompt": "  "}, "prompt"),
        ("generator", {"type": None}, "must be 'llm' or 'direct'"),
        ("generator", {"type": "unknown"}, "must be 'llm' or 'direct'"),
        (
            "generator",
            {"criteria": {"true": "Relevant", "false": "Not relevant"}},
            "only supported for direct",
        ),
        ("reranker", {"decision_model": "gpt-5"}, "Jev"),
        ("reranker", {"decision_weight": -1}, "decision_weight"),
        ("reranker", {"k": 0}, "reranker.k"),
        ("reranker", {"state_format": "{missing}"}, "state_format"),
    ],
)
def test_invalid_configuration_is_rejected(section, updates, error):
    overrides = {f"{section}_overrides": updates}
    with pytest.raises(ValueError, match=error):
        BagOfDecisionsStrategy.build(_params(**overrides), corpus=_corpus())
