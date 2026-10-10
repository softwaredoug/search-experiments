from __future__ import annotations

from dataclasses import dataclass

from exps.agentic.conditions import judging
from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
    ranked_doc_ids,
)


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

        grades = judging._oracle_grade_results(
            query=context.query,
            ranked_doc_ids=ranked_doc_ids(context.response),
            judgments=context.judgments,
            corpus=context.corpus,
            lookup=context.lookup,
        )
        if judging._judge_is_passing(grades):
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
