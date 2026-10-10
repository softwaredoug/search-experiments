from __future__ import annotations

from dataclasses import dataclass

from cheat_at_search.agent.openai_agent import OpenAIAgent

from exps.agentic.conditions import judging
from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
    ranked_doc_ids,
)
from exps.agentic.conditions.judging import (
    PASSING_EMOJI,
    GradedSearchResult,
    LLMJudgeResponse,
)


def _run_llm_judge(
    *,
    query: str,
    corpus,
    lookup: dict | None,
    ranked_doc_ids: list[str],
    model: str,
    reasoning: str,
    judge_prompt: str,
    logger,
    images: bool = False,
) -> list[GradedSearchResult]:
    results_block = judging._render_results_for_judge(
        corpus=corpus,
        ranked_doc_ids=ranked_doc_ids,
        lookup=lookup,
    )
    prompt = judge_prompt.format(query=query, results=results_block)
    judge_agent = OpenAIAgent(
        tools=[],
        model=f"openai/{model}" if "/" not in model else model,
        response_model=LLMJudgeResponse,
        reasoning_level=reasoning,
        process_images=images,
    )
    response, _, _ = judge_agent.chat(
        inputs=[{"role": "user", "content": prompt}],
        agent_state=None,
        logger=logger,
    )
    parsed = getattr(response, "output_parsed", None)
    if parsed is None:
        return []
    return list(parsed.graded_results or [])


def _is_passing(grades: list[GradedSearchResult]) -> bool:
    return bool(grades) and all(item.emoji == PASSING_EMOJI for item in grades)


@dataclass
class LLMJudgeRelevance(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> LLMJudgeRelevance:
        if kind != "validator":
            raise ValueError("llm_judge_relevance is only supported for validators.")
        for key in ("model", "reasoning", "judge_prompt"):
            if key not in params:
                raise ValueError(f"Condition 'llm_judge_relevance' requires params.{key}.")
        params.setdefault("max_runs", 2)
        if int(params["max_runs"]) <= 0:
            raise ValueError("Condition 'llm_judge_relevance' requires params.max_runs > 0.")
        return cls(
            name="llm_judge_relevance", prompt=prompt, params=params, kind=kind
        )

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        params = self.params
        if context.agent_state is not None:
            run_key = "llm_judge_runs"
            context.agent_state[run_key] = context.agent_state.get(run_key, 0) + 1
            judge_runs = context.agent_state[run_key]
        else:
            judge_runs = context.num_loops

        grades = _run_llm_judge(
            query=context.query,
            corpus=context.corpus,
            lookup=context.lookup,
            ranked_doc_ids=ranked_doc_ids(context.response),
            model=str(params["model"]),
            reasoning=str(params["reasoning"]),
            judge_prompt=str(params["judge_prompt"]),
            logger=context.logger,
            images=context.images,
        )
        if _is_passing(grades):
            return ConditionResult.success()
        if judge_runs >= int(params.get("max_runs", 2)):
            return ConditionResult.success()

        eval_block = "\n".join(
            f"{idx}. {item.emoji} {item.title} (ID: {item.doc_id})"
            for idx, item in enumerate(grades, start=1)
        )
        return self.feedback(
            f"{self.prompt}\n\nLLM evaluations:\n\n{eval_block}\n\n"
            "System reminder: return DOC IDs ranked best to worst."
        )
