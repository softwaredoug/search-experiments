from __future__ import annotations

import math
from collections.abc import Mapping

from cheat_at_search.enrich import AutoDecisionEnricher

from exps.judge.engines.common import PairwiseJudgeResult
from exps.judge.engines.jev_pointwise import _jev_model_name


class JevPairwiseJudgeEngine:
    """Use a Jev Choice to select the more relevant of two documents."""

    def __init__(
        self,
        *,
        model: str,
        explicit_neither: bool = False,
        no_cache: bool = False,
    ):
        if not isinstance(explicit_neither, bool):
            raise ValueError("strategy.params.explicit_neither must be a boolean.")
        self.model = _jev_model_name(model)
        self.explicit_neither = explicit_neither
        criteria = {
            "lhs": "The left-hand document is more relevant to the query.",
            "rhs": "The right-hand document is more relevant to the query.",
        }
        if explicit_neither:
            criteria["neither"] = "Neither document is more relevant than the other."
        self.criteria = criteria
        self.enricher = AutoDecisionEnricher(
            choices=self.criteria,
            model=f"jev/{self.model}",
        )
        if no_cache:
            self.enricher.delete_cache()

    def judge(self, *, state: str, question: str, **_kwargs) -> PairwiseJudgeResult:
        if not isinstance(state, str) or not state.strip():
            raise ValueError("Jev judge state must be a non-empty string.")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("Judge question must be a non-empty string.")
        probabilities, _confidence = self.enricher.score(
            state=state,
            question=question,
        )
        if not isinstance(probabilities, Mapping):
            raise ValueError("Jev returned invalid pairwise probabilities.")
        unexpected = set(probabilities) - set(self.criteria)
        if unexpected:
            raise ValueError(
                f"Jev returned an invalid pairwise outcome: {sorted(unexpected)[0]!r}."
            )
        for probability in probabilities.values():
            if (
                isinstance(probability, bool)
                or not isinstance(probability, (int, float))
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Jev returned an invalid probability for its choice.")
        allowed_probabilities = {
            choice: probability
            for choice, probability in probabilities.items()
            if choice in self.criteria
        }
        if not allowed_probabilities:
            outcome = next(iter(probabilities), None)
            raise ValueError(f"Jev returned an invalid pairwise outcome: {outcome!r}.")
        outcome = max(allowed_probabilities, key=allowed_probabilities.get)
        probability = float(allowed_probabilities[outcome])
        return PairwiseJudgeResult(outcome=outcome, probability=probability)
