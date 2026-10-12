from __future__ import annotations

import re
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from exps.runners.run import RunParams, run_benchmark


class ScriptedPairwiseEnricher:
    instances = []

    def __init__(self, *, model, system_prompt, response_model):
        self.model = model
        self.system_prompt = system_prompt
        self.response_model = response_model
        self.prompts = []
        self.enricher = self
        type(self).instances.append(self)

    def enrich(self, prompt):
        self.prompts.append(prompt)
        lhs_grade, rhs_grade = map(
            int,
            re.search(r"LHS: grade=(\d+)\s+RHS: grade=(\d+)", prompt).groups(),
        )
        winner = "lhs" if lhs_grade > rhs_grade else "rhs" if rhs_grade > lhs_grade else "neither"
        return self.response_model(winner=winner)


class ScriptedPairwiseJevClient:
    instances = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        self.calls = []
        type(self).instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        lhs_grade, rhs_grade = map(
            int,
            re.search(r"LHS: grade=(\d+)\s+RHS: grade=(\d+)", state).groups(),
        )
        winner = "lhs" if lhs_grade > rhs_grade else "rhs" if rhs_grade > lhs_grade else "neither"
        return SimpleNamespace(
            choices={
                "winner": SimpleNamespace(
                    choice=winner,
                    probabilities={winner: 0.99},
                )
            }
        )


def _dataset():
    grades = [2, 1, 0]
    corpus = pd.DataFrame(
        {
            "doc_id": list(range(len(grades))),
            "title": [f"Document {grade}" for grade in grades],
            "description": [f"grade={grade}" for grade in grades],
        }
    )
    judgments = pd.DataFrame(
        {
            "query_id": [1] * len(grades),
            "query": ["find relevant documents"] * len(grades),
            "doc_id": list(range(len(grades))),
            "grade": grades,
        }
    )
    return SimpleNamespace(corpus=corpus, judgments=judgments)


def _write_config(path, model):
    path.write_text(
        f"""
strategy:
  name: pairwise_judge_fixture
  type: llm_pairwise_judge
  params:
    model: {model}
    system_prompt: Compare the two documents carefully.
    prompt: |
      Query: {{query}}
      LHS: {{description_lhs}}
      RHS: {{description_rhs}}
    question: Which document is more relevant?
    to_pointwise: copeland
    sample_size: 3
    sampling_seed: 42
    double_check: false
""".lstrip(),
        encoding="utf-8",
    )


def test_pairwise_openai_strategy_runs_through_benchmark(tmp_path):
    ScriptedPairwiseEnricher.instances = []
    config_path = tmp_path / "pairwise.yml"
    _write_config(config_path, "gpt-5-mini")

    with (
        patch("exps.runners.run.get_dataset", return_value=_dataset()),
        patch(
            "exps.judge.engines.llm_pairwise.AutoEnricher",
            ScriptedPairwiseEnricher,
        ),
    ):
        result = run_benchmark(
            RunParams(
                strategy_path=str(config_path),
                dataset="doug_blog",
                num_queries=1,
                no_cache=True,
            )
        )

    assert result.strategy_name == "pairwise_judge_fixture"
    assert result.graded is not None
    assert len(result.graded) == 3
    assert result.graded["agreement"].unique().tolist() == [1.0]
    assert result.graded["pairwise_agreement"].unique().tolist() == [1.0]
    assert result.summary["mean_agreement"] == 1.0
    assert result.summary["mean_pairwise_agreement"] == 1.0
    judge = ScriptedPairwiseEnricher.instances[0]
    assert len(judge.prompts) == 3
    assert all("Query: find relevant documents" in prompt for prompt in judge.prompts)
    assert all("Which document is more relevant?" in prompt for prompt in judge.prompts)
    assert all("LHS: grade=" in prompt and "RHS: grade=" in prompt for prompt in judge.prompts)


def test_pairwise_jev_strategy_uses_choice_api_and_threshold(tmp_path):
    ScriptedPairwiseJevClient.instances = []
    config_path = tmp_path / "jev_pairwise.yml"
    _write_config(config_path, "jev/jev-latest")
    config_text = config_path.read_text(encoding="utf-8").replace(
        "    double_check: false\n", "    double_check: false\n    confidence_threshold: 0.7\n"
    )
    config_path.write_text(config_text, encoding="utf-8")

    with (
        patch("exps.runners.run.get_dataset", return_value=_dataset()),
        patch(
            "exps.judge.engines.jev_pairwise.TypeSafeClient",
            ScriptedPairwiseJevClient,
        ),
        patch("exps.judge.engines.jev_pairwise.key_for_provider", return_value="fake-key"),
    ):
        result = run_benchmark(
            RunParams(
                strategy_path=str(config_path),
                dataset="doug_blog",
                num_queries=1,
                no_cache=True,
            )
        )

    assert len(result.graded) == 3
    assert result.graded["agreement"].unique().tolist() == [1.0]
    client = ScriptedPairwiseJevClient.instances[0]
    assert client.model == "jev-latest"
    assert len(client.calls) == 3
    state, questions = client.calls[0]
    assert "Query: find relevant documents" in state
    assert "LHS: grade=" in state and "RHS: grade=" in state
    assert "Compare these WANDS" not in state
    assert questions["winner"].instructions == "Which document is more relevant?"
    assert set(questions["winner"].criteria) == {"lhs", "rhs"}
