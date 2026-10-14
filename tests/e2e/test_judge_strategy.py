from __future__ import annotations

import csv
import re
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from exps.runner import _write_summary_csv
from exps.runners.run import RunParams, run_benchmark


class ScriptedJudgeEnricher:
    instances = []

    def __init__(self, *, model, system_prompt, response_model):
        self.model = model
        self.system_prompt = system_prompt
        self.response_model = response_model
        self.prompts = []
        self.enricher = self
        self.__class__.instances.append(self)

    def enrich(self, prompt):
        self.prompts.append(prompt)
        grade = int(re.search(r"Description: grade=(\d+)", prompt).group(1))
        return self.response_model(grade=grade)


class ScriptedJevClient:
    instances = []

    def __init__(self, *, api_key, model):
        self.api_key = api_key
        self.model = model
        self.calls = []
        type(self).instances.append(self)

    def system_one(self, *, state, questions):
        self.calls.append((state, questions))
        grade = re.search(r"Description: grade=(\d+)", state).group(1)
        return SimpleNamespace(
            choices={
                "grade": SimpleNamespace(
                    choice=grade,
                    probabilities={grade: 0.99},
                )
            }
        )


def test_judge_strategy_samples_stratifies_and_reports_agreement(tmp_path):
    ScriptedJudgeEnricher.instances = []
    grades = [2] * 4 + [1] * 4 + [0] * 4
    corpus = pd.DataFrame(
        {
            "doc_id": list(range(len(grades))),
            "title": [f"Document {doc_id}" for doc_id in range(len(grades))],
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
    dataset = SimpleNamespace(corpus=corpus, judgments=judgments)
    config_path = tmp_path / "judge.yml"
    config_path.write_text(
        """
strategy:
  name: judge_fixture
  type: llm_pontwise_judge
  params:
    model: gpt-5-mini
    system_prompt: Grade documents consistently.
    prompt: |
      Query: {query}
      Title: {title}
      Description: {description}
    question: Select the numeric grade that best matches the document's relevance to the query.
    rubric:
      2: relevant
      1: partially relevant
      0: irrelevant
    sample_size: 10
    sampling_seed: 42
""".lstrip(),
        encoding="utf-8",
    )

    with (
        patch("exps.runners.run.get_dataset", return_value=dataset),
        patch("exps.judge.engines.llm_pointwise.AutoEnricher", ScriptedJudgeEnricher),
    ):
        result = run_benchmark(
            RunParams(
                strategy_path=str(config_path),
                dataset="doug_blog",
                num_queries=1,
                no_cache=True,
            )
        )

    assert result.strategy_name == "judge_fixture"
    assert result.graded is not None
    assert len(result.graded) == 9
    assert sorted(result.graded["grade"].value_counts().tolist()) == [3, 3, 3]
    assert result.graded["agreement"].unique().tolist() == [1.0]
    assert result.graded["pairwise_agreement"].unique().tolist() == [1.0]
    assert result.summary["mean_agreement"] == 1.0
    assert result.summary["mean_pairwise_agreement"] == 1.0

    report_path = tmp_path / "wands_judge_summary.csv"
    _write_summary_csv(str(report_path), dataset="wands", result=result)
    with report_path.open(encoding="utf-8", newline="") as handle:
        report_row = next(csv.DictReader(handle))
    assert report_row["mean_agreement"] == "1.0"
    assert report_row["mean_pairwise_agreement"] == "1.0"

    judge = ScriptedJudgeEnricher.instances[0]
    assert len(judge.prompts) == 9
    assert all("Query: find relevant documents" in prompt for prompt in judge.prompts)
    assert all("Select the numeric grade" in prompt for prompt in judge.prompts)
    assert all("2: relevant" in prompt and "0: irrelevant" in prompt for prompt in judge.prompts)
    assert all(prompt.count("Select the numeric grade") == 1 for prompt in judge.prompts)


def test_jev_judge_strategy_uses_choice_api_and_reports_grades(tmp_path):
    ScriptedJevClient.instances = []
    grades = [2] * 4 + [1] * 4 + [0] * 4
    corpus = pd.DataFrame(
        {
            "doc_id": list(range(len(grades))),
            "title": [f"Document {doc_id}" for doc_id in range(len(grades))],
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
    dataset = SimpleNamespace(corpus=corpus, judgments=judgments)
    config_path = tmp_path / "jev_judge.yml"
    config_path.write_text(
        """
strategy:
  name: jev_judge_fixture
  type: llm_pontwise_judge
  params:
    model: jev/jev-latest
    prompt: |
      Query: {query}
      Title: {title}
      Description: {description}
    question: Select the numeric relevance grade.
    rubric:
      2: relevant
      1: partially relevant
      0: irrelevant
    sample_size: 10
    sampling_seed: 42
""".lstrip(),
        encoding="utf-8",
    )

    with (
        patch("exps.runners.run.get_dataset", return_value=dataset),
        patch("exps.judge.engines.jev_pointwise.TypeSafeClient", ScriptedJevClient),
        patch("exps.judge.engines.jev_pointwise.key_for_provider", return_value="fake-key"),
    ):
        result = run_benchmark(
            RunParams(
                strategy_path=str(config_path),
                dataset="doug_blog",
                num_queries=1,
                no_cache=True,
            )
        )

    assert len(result.graded) == 9
    assert result.graded["agreement"].unique().tolist() == [1.0]
    assert result.graded["pairwise_agreement"].unique().tolist() == [1.0]
    client = ScriptedJevClient.instances[0]
    assert client.model == "jev-latest"
    assert len(client.calls) == 9
    state, questions = client.calls[0]
    assert "Query: find relevant documents" in state
    assert "grade=" in state
    assert "Select the numeric relevance grade." == questions["grade"].instructions
    assert questions["grade"].criteria == {
        "2": "relevant",
        "1": "partially relevant",
        "0": "irrelevant",
    }


def test_openai_judge_respects_question_and_rubric_prompt_placeholders(tmp_path):
    ScriptedJudgeEnricher.instances = []
    grades = [2, 1, 0]
    corpus = pd.DataFrame(
        {
            "doc_id": list(range(len(grades))),
            "title": [f"Document {doc_id}" for doc_id in range(len(grades))],
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
    dataset = SimpleNamespace(corpus=corpus, judgments=judgments)
    config_path = tmp_path / "judge_with_placeholders.yml"
    question = "Select the numeric grade for this document."
    config_path.write_text(
        f"""
strategy:
  name: judge_placeholder_fixture
  type: llm_pontwise_judge
  params:
    model: openai/gpt-5-mini
    prompt: |
      Query: {{query}}
      Description: {{description}}
      Question: {{question}}
      Rubric:
      {{rubric}}
    question: {question}
    rubric:
      2: relevant
      1: partially relevant
      0: irrelevant
    sample_size: 3
""".lstrip(),
        encoding="utf-8",
    )

    with (
        patch("exps.runners.run.get_dataset", return_value=dataset),
        patch("exps.judge.engines.llm_pointwise.AutoEnricher", ScriptedJudgeEnricher),
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
    assert len(ScriptedJudgeEnricher.instances) == 1
    prompts = ScriptedJudgeEnricher.instances[0].prompts
    assert len(prompts) == 3
    assert all(prompt.count(question) == 1 for prompt in prompts)
    assert all(prompt.count("2: relevant") == 1 for prompt in prompts)
