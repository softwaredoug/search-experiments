from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from exps.agentic.conditions import judging
from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
    ranked_doc_ids,
)
from exps.agentic.conditions.judging import PASSING_EMOJI, GradedSearchResult


def _oracle_emojis_for_grades(grades: list) -> tuple[dict, list]:
    if not grades:
        raise ValueError("Oracle validator requires at least one grade label.")

    def grade_key(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    sorted_grades = sorted(grades, key=grade_key)
    if len(sorted_grades) == 2:
        return {sorted_grades[0]: "😞", sorted_grades[1]: "😃"}, sorted_grades
    if len(sorted_grades) == 3:
        return {
            sorted_grades[0]: "😞",
            sorted_grades[1]: "😐",
            sorted_grades[2]: "😃",
        }, sorted_grades
    if len(sorted_grades) == 4:
        return {
            sorted_grades[0]: "😞",
            sorted_grades[1]: "😐",
            sorted_grades[2]: "😃",
            sorted_grades[3]: "🤩",
        }, sorted_grades
    raise ValueError("Oracle validator supports only 2, 3, or 4 unique grade labels.")


def _oracle_grade_results(
    *,
    query: str,
    ranked_doc_ids: list[str],
    judgments,
    corpus,
    lookup: dict | None,
) -> list[GradedSearchResult]:
    if judgments is None:
        raise ValueError("Oracle validator requires judgments.")
    if "grade" not in judgments.columns:
        raise ValueError("Oracle validator requires judgments with a 'grade' column.")
    if "query" not in judgments.columns:
        raise ValueError("Oracle validator requires judgments with a 'query' column.")

    query_judgments = judgments[judgments["query"] == query]
    grades = judgments["grade"].dropna().unique().tolist()
    emoji_map, ordered_grades = _oracle_emojis_for_grades(grades)
    negative_emoji = emoji_map[ordered_grades[0]]

    grade_by_doc: dict[str, Any] = {}
    if not query_judgments.empty:
        grade_order = {grade: idx for idx, grade in enumerate(ordered_grades)}
        for doc_id, group in query_judgments.groupby("doc_id"):
            grade_by_doc[str(doc_id)] = max(
                group["grade"], key=lambda value: grade_order.get(value, -1)
            )

    graded_results = []
    for doc_id in ranked_doc_ids:
        grade = grade_by_doc.get(str(doc_id))
        emoji = emoji_map.get(grade, negative_emoji)
        row, _ = judging._judge_row_for_doc_id(
            corpus=corpus,
            doc_id=doc_id,
            lookup=lookup,
        )
        title = str(row.get("title", "Sample")) if row is not None else "Sample"
        graded_results.append(
            GradedSearchResult(emoji=emoji, title=title, doc_id=str(doc_id))
        )
    return graded_results


def _is_passing(grades: list[GradedSearchResult]) -> bool:
    return bool(grades) and all(item.emoji == PASSING_EMOJI for item in grades)


@dataclass
class OracleValidator(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> OracleValidator:
        name = "oracle"
        if kind != "validator":
            raise ValueError("oracle is only supported for validators.")
        params.setdefault("max_runs", 2)
        if int(params["max_runs"]) <= 0:
            raise ValueError("Condition 'oracle' requires params.max_runs > 0.")
        return cls(name=name, prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        params = self.params
        if context.agent_state is not None:
            run_key = "oracle_runs"
            context.agent_state[run_key] = context.agent_state.get(run_key, 0) + 1
            oracle_runs = context.agent_state[run_key]
        else:
            oracle_runs = context.num_loops

        grades = _oracle_grade_results(
            query=context.query,
            ranked_doc_ids=ranked_doc_ids(context.response),
            judgments=context.judgments,
            corpus=context.corpus,
            lookup=context.lookup,
        )
        if _is_passing(grades):
            return ConditionResult.success()
        if oracle_runs >= int(params.get("max_runs", 2)):
            return ConditionResult.success()

        eval_block = "\n".join(
            f"{idx}. {item.emoji} {item.title} (ID: {item.doc_id})"
            for idx, item in enumerate(grades, start=1)
        )
        return self.feedback(
            f"{self.prompt}\n\nOracle evaluations:\n\n{eval_block}\n\n"
            "System reminder: return DOC IDs ranked best to worst."
        )
