from __future__ import annotations

import hashlib
import json
import string
from typing import Any


class DirectDecisionGenerator:
    """Format one configured yes/no question for each search query."""

    def __init__(self, *, question: str):
        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                "decision_engine.generator.question must be a non-empty template "
                "containing {query}."
            )
        self.question_template = question.strip()
        try:
            fields = [
                field_name
                for _, field_name, _, _ in string.Formatter().parse(
                    self.question_template
                )
                if field_name
            ]
        except ValueError as exc:
            raise ValueError(
                "decision_engine.generator.question must be a valid template "
                "containing {query}."
            ) from exc
        if "query" not in fields or any(field != "query" for field in fields):
            raise ValueError(
                "decision_engine.generator.question must contain {query} and "
                "may not use other fields."
            )

    def generate(self, query: str) -> list[str]:
        try:
            question = self.question_template.format(query=query)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "Could not format decision_engine.generator.question with {query}."
            ) from exc
        return [question]

    @property
    def cache_key(self) -> str:
        payload: dict[str, Any] = {
            "type": "direct_decision_generator",
            "question": self.question_template,
        }
        serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.md5(serialized).hexdigest()
