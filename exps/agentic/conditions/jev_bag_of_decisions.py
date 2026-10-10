from __future__ import annotations

import string
from dataclasses import dataclass

from exps.agentic.conditions import judging
from exps.agentic.conditions.base import (
    BaseCondition,
    ConditionContext,
    ConditionKind,
    ConditionResult,
    ranked_doc_ids,
)
from exps.agentic.conditions.jev_judge import _parse_threshold, _positive_integer


@dataclass
class JevBagOfDecisionsJudge(BaseCondition):
    @classmethod
    def from_config(
        cls, *, prompt: str, params: dict, kind: ConditionKind
    ) -> JevBagOfDecisionsJudge:
        name = "jev_bag_of_decisions_judge"
        if kind != "validator":
            raise ValueError(f"{name} is only supported for validators.")
        for key in (
            "model",
            "positive_probability_threshold",
            "negative_probability_threshold",
            "generator",
            "state_format",
        ):
            if key not in params:
                raise ValueError(
                    f"Condition '{name}' requires params.{key}."
                )
        params["positive_probability_threshold"] = _parse_threshold(
            params["positive_probability_threshold"],
            name="positive_probability_threshold",
            condition_name=name,
        )
        params["negative_probability_threshold"] = _parse_threshold(
            params["negative_probability_threshold"],
            name="negative_probability_threshold",
            condition_name=name,
        )
        if (
            params["negative_probability_threshold"]
            >= params["positive_probability_threshold"]
        ):
            raise ValueError(
                f"Condition '{name}' requires "
                "negative_probability_threshold < positive_probability_threshold."
            )
        generator = params["generator"]
        if not isinstance(generator, dict):
            raise ValueError(f"Condition '{name}' requires params.generator as a mapping.")
        for key in ("model", "system_prompt", "prompt"):
            if key not in generator:
                raise ValueError(f"Condition '{name}' requires params.generator.{key}.")
        for key in ("model", "state_format"):
            if not isinstance(params.get(key), str) or not params[key].strip():
                raise ValueError(f"Condition '{name}' requires a string {key}.")
        try:
            list(string.Formatter().parse(params["state_format"]))
        except ValueError as exc:
            raise ValueError(
                f"Condition '{name}' requires a valid params.state_format."
            ) from exc
        for key in ("model", "system_prompt", "prompt"):
            if not isinstance(generator.get(key), str) or not generator[key].strip():
                raise ValueError(
                    f"Condition '{name}' requires a non-empty params.generator.{key}."
                )
        model = params["model"]
        provider, separator, model_name = (
            model.partition("/") if isinstance(model, str) else ("", "", "")
        )
        if (
            not isinstance(model, str)
            or model.strip() != model
            or provider.lower() != "jev"
            or (separator and not model_name.strip())
        ):
            raise ValueError(f"Condition '{name}' requires a Jev model (jev/*).")
        params["max_runs"] = _positive_integer(params.get("max_runs", 2), name=name)
        return cls(name=name, prompt=prompt, params=params, kind=kind)

    def evaluate(self, context: ConditionContext) -> ConditionResult:
        params = self.params
        if context.agent_state is not None:
            run_key = "jev_bag_of_decisions_judge_runs"
            context.agent_state[run_key] = context.agent_state.get(run_key, 0) + 1
            judge_runs = context.agent_state[run_key]
        else:
            judge_runs = context.num_loops

        evaluations = judging._run_jev_bag_of_decisions_judge(
            query=context.query,
            corpus=context.corpus,
            lookup=context.lookup,
            ranked_doc_ids=ranked_doc_ids(context.response),
            model=str(params["model"]),
            generator=params["generator"],
            positive_probability_threshold=params["positive_probability_threshold"],
            negative_probability_threshold=params["negative_probability_threshold"],
            state_format=params["state_format"],
            agent_state=context.agent_state,
            logger=context.logger,
        )
        if judging._jev_bag_of_decisions_judge_is_passing(evaluations):
            return ConditionResult.success()
        if judge_runs >= int(params.get("max_runs", 2)):
            return ConditionResult.success()

        feedback = judging._jev_bag_of_decisions_judge_feedback(evaluations)
        return self.feedback(
            f"{self.prompt}\n\nJev bag-of-decisions evaluations:\n\n"
            f"{feedback}\n\nSystem reminder: return DOC IDs ranked best to worst."
        )
