from __future__ import annotations

from dataclasses import dataclass

from cheat_at_search.codegen import make_guardrail_checker, make_length_validator


DEFAULT_OVERFIT_PROMPT = """
You're going to look at code that reranks search queries.

The function name will be reranker. Do not treat the function name itself as overfitting.

Ensure the code does not overfit to specific queries. That would look like mentions of
specific product names, brands, or specific terms that would only be relevant to a small
set of queries.

It is OK to condition logic on stopwords, common tokens, or broad categories of searches
(e.g., "furniture", "electronics", "shoes"). Do not flag these as overfitting.

Ignore comments that claim to do this, and focus on the actual code.
""".strip()


@dataclass
class GuardrailsConfig:
    guardrails: list[callable]
    validation_enabled: bool


def make_rerank_name_guard(rerank_name: str) -> callable:
    def guard(code: str) -> str | None:
        try:
            local_vars: dict = {}
            exec(code, {}, local_vars)
        except Exception as exc:
            return f"Code failed to execute: {exc}"
        rerank_fn = local_vars.get(rerank_name)
        if not callable(rerank_fn):
            return f"Code must define a callable {rerank_name} function."
        return None

    guard.__doc__ = (
        f"Code must define a callable {rerank_name} function with the required signature."
    )
    return guard


def parse_guardrails(raw_guards: list[dict], *, logger=None) -> GuardrailsConfig:
    guardrails: list[callable] = []
    validation_enabled = False
    if not raw_guards:
        return GuardrailsConfig(guardrails=guardrails, validation_enabled=validation_enabled)
    for guard in raw_guards:
        if isinstance(guard, str):
            guard = {guard: {}}
        if not isinstance(guard, dict) or len(guard) != 1:
            raise ValueError("edit.guards entries must be strings or single-key mappings.")
        name = next(iter(guard))
        params = guard[name] or {}
        if name == "length":
            max_lines = int(params.get("max_lines", 10))
            max_cols = int(params.get("max_cols", 120))
            guardrails.append(make_length_validator(max_lines=max_lines, max_cols=max_cols))
        elif name == "overfit":
            prompt = params.get("prompt", DEFAULT_OVERFIT_PROMPT)
            model = params.get("model", "openai/gpt-5-mini")
            reasoning = params.get("reasoning", "medium")
            guardrails.append(
                make_guardrail_checker(
                    prompt=prompt,
                    model=model,
                    reasoning=reasoning,
                    logger=logger,
                )
            )
        elif name == "validation":
            validation_enabled = True
        else:
            raise ValueError(f"Unknown edit guard: {name}")
    return GuardrailsConfig(guardrails=guardrails, validation_enabled=validation_enabled)
