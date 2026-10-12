from __future__ import annotations

import hashlib
import json
import random
import string
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
from cheat_at_search.strategy import SearchStrategy

from exps.judge.engines.common import grade_value, normalize_rubric, rubric_text
from exps.judge.engines.jev_pointwise import JevJudgeEngine
from exps.judge.engines.llm_pointwise import LLMJudgeEngine


@dataclass(frozen=True)
class JudgeRequest:
    prompt: str
    question: str
    prompt_has_question: bool
    prompt_has_rubric: bool


def _template_fields(template: str) -> set[str]:
    try:
        return {
            field_name.split(".", 1)[0].split("[", 1)[0]
            for _, field_name, _, _ in string.Formatter().parse(template)
            if field_name
        }
    except ValueError as exc:
        raise ValueError("Judge prompt and question must be valid format strings.") from exc


class PointwiseJudgeStrategy(SearchStrategy):
    """Sample judged query-document pairs and rank them with a pointwise judge."""

    _type = "llm_pontwise_judge"

    @classmethod
    def build(
        cls,
        params: dict,
        *,
        corpus,
        workers: int = 1,
        judgments: pd.DataFrame | None = None,
        no_cache: bool = False,
        **_kwargs,
    ):
        if judgments is None:
            raise ValueError("The judge strategy requires dataset judgments.")
        return cls(
            corpus,
            judgments=judgments,
            workers=workers,
            no_cache=no_cache,
            **params,
        )

    def __init__(
        self,
        corpus,
        *,
        judgments: pd.DataFrame,
        model: str = "gpt-5-mini",
        prompt: str,
        question: str,
        rubric: Mapping,
        system_prompt: str = "You are a careful search relevance judge.",
        sample_size: int = 10,
        sampling_seed: int = 42,
        workers: int = 1,
        no_cache: bool = False,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("strategy.params.prompt must be a non-empty string.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("strategy.params.question must be a non-empty string.")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("strategy.params.system_prompt must be a non-empty string.")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("strategy.params.model must be a non-empty string.")
        if isinstance(sample_size, bool) or not isinstance(sample_size, int):
            raise ValueError("strategy.params.sample_size must be an integer from 1 to 10.")
        if not 1 <= sample_size <= 10:
            raise ValueError("strategy.params.sample_size must be an integer from 1 to 10.")
        if isinstance(sampling_seed, bool) or not isinstance(sampling_seed, int):
            raise ValueError("strategy.params.sampling_seed must be an integer.")
        if "query" not in judgments or "doc_id" not in judgments:
            raise ValueError("Judgments must contain 'query' and 'doc_id' columns.")
        grade_col = next(
            (
                column
                for column in ("grade", "relevance", "rel", "label", "score")
                if column in judgments.columns
            ),
            None,
        )
        if grade_col is None:
            raise ValueError("Judgments must contain a numeric grade column.")
        if "doc_id" not in corpus:
            raise ValueError("The corpus must contain a 'doc_id' column.")

        self.rubric = normalize_rubric(rubric)
        self.corpus = corpus
        self.model = model.strip()
        self.prompt_template = prompt
        self.question_template = question
        self.system_prompt = system_prompt
        self.sample_size = sample_size
        self.sampling_seed = sampling_seed
        self.no_cache = no_cache
        self.grade_column = grade_col
        self._prompt_fields = _template_fields(prompt)
        self._question_fields = _template_fields(question)
        available_fields = set(corpus.columns) | {"query", "rubric", "question"}
        missing_fields = sorted(
            (self._prompt_fields | self._question_fields) - available_fields
        )
        if missing_fields:
            raise ValueError(
                "Judge prompt placeholders must be 'query', 'rubric', 'question', "
                "or corpus columns; missing: " + ", ".join(missing_fields)
            )

        provider, separator, model_name = self.model.partition("/")
        provider = provider.lower()
        if separator and (provider not in {"openai", "jev"} or not model_name.strip()):
            raise ValueError(
                "strategy.params.model must be an OpenAI model or a Jev model (jev/*)."
            )
        self.engine_type = JevJudgeEngine if provider == "jev" else LLMJudgeEngine

        self._doc_indices = {
            doc_id: index
            for index, doc_id in enumerate(corpus["doc_id"].tolist())
            if not pd.isna(doc_id)
        }
        self._samples = self._sample_judgments(judgments)
        all_grades = {
            grade_value(value)
            for value in judgments[grade_col].dropna().unique().tolist()
        }
        if not all_grades.issubset(self.rubric):
            missing = sorted(all_grades - set(self.rubric))
            raise ValueError(f"Rubric is missing judgment grade(s): {missing}.")
        if self.engine_type is JevJudgeEngine:
            self.engine = JevJudgeEngine(model=self.model, rubric=self.rubric)
        else:
            self.engine = LLMJudgeEngine(
                model=self.model,
                system_prompt=self.system_prompt,
                rubric=self.rubric,
                no_cache=no_cache,
            )
        super().__init__(corpus, workers=workers)

    def _sample_judgments(
        self, judgments: pd.DataFrame
    ) -> dict[str, list[tuple[int, float]]]:
        samples: dict[str, list[tuple[int, float]]] = {}
        available = judgments.dropna(subset=["query", "doc_id", self.grade_column])
        for query, query_rows in available.groupby("query", sort=False):
            by_grade: dict[float, list[tuple[int, float]]] = {}
            for _, row in query_rows.iterrows():
                index = self._doc_indices.get(row["doc_id"])
                if index is None:
                    continue
                grade = grade_value(row[self.grade_column])
                by_grade.setdefault(grade, []).append((index, grade))

            grades = sorted(by_grade)
            if not grades:
                samples[str(query)] = []
                continue
            digest = hashlib.sha256(
                f"{self.sampling_seed}\0{query}".encode("utf-8")
            ).digest()
            rng = random.Random(int.from_bytes(digest[:8], "big"))
            for rows in by_grade.values():
                rng.shuffle(rows)

            target = min(self.sample_size, sum(map(len, by_grade.values())))
            start_grade = rng.randrange(len(grades))
            allocation_order = grades[start_grade:] + grades[:start_grade]
            quotas = {
                grade: min(target // len(grades), len(by_grade[grade]))
                for grade in grades
            }
            remaining = target - sum(quotas.values())
            while remaining:
                progressed = False
                for grade in allocation_order:
                    if quotas[grade] < len(by_grade[grade]):
                        quotas[grade] += 1
                        remaining -= 1
                        progressed = True
                        if remaining == 0:
                            break
                if not progressed:
                    break

            selected = []
            for grade in grades:
                selected.extend(by_grade[grade][: quotas[grade]])
            rng.shuffle(selected)
            samples[str(query)] = selected
        return samples

    def _render_request(self, query: str, index: int) -> JudgeRequest:
        row = self.corpus.iloc[index]
        values: dict[str, Any] = {key: value for key, value in row.items()}
        values.update(query=query, rubric=rubric_text(self.rubric))
        try:
            question = self.question_template.format_map(values)
            values["question"] = question
            prompt = self.prompt_template.format_map(values)
        except (KeyError, ValueError, IndexError) as exc:
            raise ValueError(
                "Judge prompt placeholders must be 'query', 'rubric', 'question', "
                "or corpus columns."
            ) from exc
        return JudgeRequest(
            prompt=prompt,
            question=question,
            prompt_has_question="question" in self._prompt_fields,
            prompt_has_rubric="rubric" in self._prompt_fields,
        )

    def _judge(self, query: str, index: int) -> float:
        request = self._render_request(query, index)
        result = self.engine.judge(
            prompt=request.prompt,
            question=request.question,
            rubric=self.rubric,
            prompt_has_question=request.prompt_has_question,
            prompt_has_rubric=request.prompt_has_rubric,
        )
        grade = getattr(result, "grade", result)
        numeric_grade = grade_value(grade)
        if numeric_grade not in self.rubric:
            raise ValueError(f"Judge returned a grade not in the rubric: {grade!r}.")
        return numeric_grade

    def search(self, query: str, k: int = 10):
        judged = self._samples.get(str(query), [])[:k]
        scores = [self._judge(query, index) for index, _ in judged]
        ranked = sorted(zip(judged, scores), key=lambda item: (-item[1], item[0][0]))
        return [item[0][0] for item in ranked], [item[1] for item in ranked]

    @property
    def cache_key(self) -> str:
        payload = {
            "type": self._type,
            "model": self.model,
            "prompt": self.prompt_template,
            "question": self.question_template,
            "system_prompt": self.system_prompt,
            "rubric": self.rubric,
            "sample_size": self.sample_size,
            "sampling_seed": self.sampling_seed,
            "samples": self._samples,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()


# Preserve the name previously exported by exps.strategies.judge.
LLMJudgeStrategy = PointwiseJudgeStrategy
