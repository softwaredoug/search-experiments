from __future__ import annotations

from dataclasses import dataclass

from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
    ranked_doc_ids,
)


@dataclass
class ValidDocIdsValidator(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> ValidDocIdsValidator:
        if kind != "validator":
            raise ValueError("Condition 'valid_doc_ids' can only be used as a validator.")
        if params:
            raise ValueError("Condition 'valid_doc_ids' does not accept params.")
        return cls(name="valid_doc_ids", prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        if not context.lookup:
            return ConditionResult.success()

        doc_ids = [str(doc_id) for doc_id in ranked_doc_ids(context.response)]
        seen = set()
        duplicate_ids = []
        invalid_ids = []
        for doc_id in doc_ids:
            if doc_id not in context.lookup:
                invalid_ids.append(doc_id)
            if doc_id in seen and doc_id not in duplicate_ids:
                duplicate_ids.append(doc_id)
            seen.add(doc_id)

        if not invalid_ids and not duplicate_ids:
            return ConditionResult.success()

        feedback = [self.prompt]
        if invalid_ids:
            feedback.append(f"Unknown document IDs: {invalid_ids[:5]!r}.")
        if duplicate_ids:
            feedback.append(f"Repeated document IDs: {duplicate_ids[:5]!r}.")
        feedback.append(
            "Use distinct document IDs exactly as returned by the search tools."
        )
        return ConditionResult.unsatisfied(" ".join(feedback))
