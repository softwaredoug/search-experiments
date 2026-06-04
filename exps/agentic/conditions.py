from __future__ import annotations

from typing import Any, Literal

from cheat_at_search.agent.openai_agent import OpenAIAgent
from pydantic import BaseModel, Field


AllowedEmoji = Literal["😃", "😐", "😞"]


class GradedSearchResult(BaseModel):
    """A single judged search result with an emoji relevance label."""

    emoji: AllowedEmoji = Field(description="Emoji relevance label for this result.")
    title: str = Field(description="Document title for the judged result.")
    doc_id: str = Field(description="Document ID for the judged result.")


class LLMJudgeResponse(BaseModel):
    """Structured response from the LLM judge containing graded results."""

    graded_results: list[GradedSearchResult] = Field(
        default_factory=list,
        description="Ordered list of graded search results with emoji labels.",
    )


def _parse_condition_entry(entry: Any) -> tuple[str, dict]:
    if not (isinstance(entry, dict) and len(entry) == 1):
        raise ValueError("Condition entries must be single-key mappings.")
    (name, raw_params), = entry.items()
    if not isinstance(raw_params, dict):
        raise ValueError("Condition params must be a mapping.")
    return name, dict(raw_params)


def _require_prompt(condition: dict, *, kind: str) -> str:
    prompt = condition.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{kind} condition requires a non-empty prompt.")
    return prompt


def normalize_conditions(condition_config: list | None, *, kind: str) -> list[dict[str, Any]]:
    if not condition_config:
        return []
    conditions: list[dict[str, Any]] = []
    for entry in condition_config:
        name, params = _parse_condition_entry(entry)
        prompt = _require_prompt(params, kind=kind)
        params = dict(params)
        params.pop("prompt", None)
        if "params" not in params or not isinstance(params["params"], dict):
            raise ValueError(f"{kind} condition '{name}' requires params mapping.")
        params_dict = params["params"]
        if name == "iterations":
            if "iterations" not in params_dict:
                raise ValueError("Condition 'iterations' requires params.iterations.")
        elif name == "tool_calls":
            if "num_calls" not in params_dict:
                raise ValueError("Condition 'tool_calls' requires params.num_calls.")
        elif name == "num_results":
            if "min_results" not in params_dict:
                raise ValueError("Condition 'num_results' requires params.min_results.")
        elif name == "llm_judge_relevance":
            if kind != "validator":
                raise ValueError("llm_judge_relevance is only supported for validators.")
            for key in ("model", "reasoning", "judge_prompt"):
                if key not in params_dict:
                    raise ValueError(f"Condition 'llm_judge_relevance' requires params.{key}.")
            params_dict.setdefault("max_runs", 2)
            if int(params_dict["max_runs"]) <= 0:
                raise ValueError("Condition 'llm_judge_relevance' requires params.max_runs > 0.")
        else:
            raise ValueError(f"Unknown {kind} condition: {name}")
        conditions.append({"name": name, "prompt": prompt, "params": params_dict})
    return conditions


def _num_results_from_response(resp) -> int:
    if resp is None:
        return 0
    parsed = getattr(resp, "output_parsed", None)
    if parsed is None:
        return 0
    ranked = getattr(parsed, "ranked_results", None)
    if not ranked:
        return 0
    return len(ranked)


def _condition_met(condition: dict, *, num_loops: int, tool_calls: int, resp) -> bool:
    name = condition["name"]
    params = condition["params"]
    if name == "iterations":
        return num_loops >= int(params["iterations"])
    if name == "tool_calls":
        return tool_calls >= int(params["num_calls"])
    if name == "num_results":
        return _num_results_from_response(resp) >= int(params["min_results"])
    return False


def _render_results_for_judge(*, corpus, ranked_doc_ids: list[str], lookup: dict | None) -> str:
    lines = []
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        try:
            doc_id_int = int(doc_id)
        except (TypeError, ValueError):
            continue
        row = None
        if lookup is not None and doc_id_int in lookup:
            row = corpus.iloc[lookup[doc_id_int]]
        elif "doc_id" in corpus.columns:
            match = corpus[corpus["doc_id"] == doc_id_int]
            if not match.empty:
                row = match.iloc[0]
        title = ""
        description = ""
        if row is not None:
            title = str(row.get("title", ""))
            description = str(row.get("description", ""))
        if len(description) > 200:
            description = description[:197] + "..."
        lines.append(f"{idx}. {title} (ID: {doc_id_int})\n{description}")
    return "\n\n".join(lines)


def _judge_is_passing(graded_results: list[GradedSearchResult]) -> bool:
    if not graded_results:
        return False
    return all(item.emoji == "😃" for item in graded_results)


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
) -> list[GradedSearchResult]:
    results_block = _render_results_for_judge(
        corpus=corpus,
        ranked_doc_ids=ranked_doc_ids,
        lookup=lookup,
    )
    prompt = judge_prompt.format(query=query, results=results_block)
    inputs = [{"role": "user", "content": prompt}]
    judge_agent = OpenAIAgent(
        tools=[],
        model=f"openai/{model}" if "/" not in model else model,
        response_model=LLMJudgeResponse,
        reasoning_level=reasoning,
    )
    resp, _, _ = judge_agent.chat(inputs=inputs, agent_state=None, logger=logger)
    parsed = getattr(resp, "output_parsed", None)
    if parsed is None:
        return []
    return list(parsed.graded_results or [])


def evaluate_validator(
    condition: dict,
    *,
    num_loops: int,
    tool_calls: int,
    resp,
    query: str,
    corpus,
    lookup: dict | None,
    agent_state: dict | None,
    logger,
) -> bool | str:
    if condition["name"] == "llm_judge_relevance":
        ranked = getattr(resp.output_parsed, "ranked_results", None) if resp else None
        ranked_doc_ids = list(ranked or [])
        params = condition["params"]
        max_runs = int(params.get("max_runs", 2))
        if agent_state is not None:
            agent_state["llm_judge_runs"] = agent_state.get("llm_judge_runs", 0) + 1
            judge_runs = agent_state["llm_judge_runs"]
        else:
            judge_runs = num_loops
        graded_results = _run_llm_judge(
            query=query,
            corpus=corpus,
            lookup=lookup,
            ranked_doc_ids=ranked_doc_ids,
            model=str(params["model"]),
            reasoning=str(params["reasoning"]),
            judge_prompt=str(params["judge_prompt"]),
            logger=logger,
        )
        if _judge_is_passing(graded_results):
            return True
        if judge_runs >= max_runs:
            return True
        if not _judge_is_passing(graded_results):
            eval_block = "\n".join(
                f"{idx}. {item.emoji} {item.title} (ID: {item.doc_id})"
                for idx, item in enumerate(graded_results, start=1)
            )
            return (
                f"{condition['prompt']}\n\n"
                f"LLM evaluations:\n\n{eval_block}\n\n"
                "System reminder: return DOC IDs ranked best to worst."
            )
        return True
    if _condition_met(condition, num_loops=num_loops, tool_calls=tool_calls, resp=resp):
        return True
    return condition["prompt"]


def evaluate_stopper(
    condition: dict,
    *,
    num_loops: int,
    tool_calls: int,
    resp,
) -> bool | str:
    if _condition_met(condition, num_loops=num_loops, tool_calls=tool_calls, resp=resp):
        return True
    return condition["prompt"]
