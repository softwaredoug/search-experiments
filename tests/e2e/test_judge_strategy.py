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
      Rubric: {rubric}
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
        patch("exps.strategies.judge.AutoEnricher", ScriptedJudgeEnricher),
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
    assert len(result.graded) == 10
    assert sorted(result.graded["grade"].value_counts().tolist()) == [3, 3, 4]
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
    assert len(judge.prompts) == 10
    assert all("Query: find relevant documents" in prompt for prompt in judge.prompts)
    assert all("2: relevant" in prompt and "0: irrelevant" in prompt for prompt in judge.prompts)
