from __future__ import annotations

import hashlib
import json
import math
from time import monotonic
from typing import Any, Literal

from cheat_at_search.agent.openai_agent import OpenAIAgent
from pydantic import BaseModel, Field
from cheat_at_search.data_dir import key_for_provider
from typesafe_sdk import Choice, Noul, NoulCriteria, RetryPolicy, TypeSafeClient

from exps.bag_of_decisions.decision_generator import DecisionGenerator
from exps.bag_of_decisions.decision_question import DecisionQuestion
from exps.agentic.tracing import set_trace_stage, trace_event


AllowedEmoji = Literal["🤩", "😃", "😐", "😞"]


class _TracingRetryPolicy(RetryPolicy):
    """Keep the TypeSafe defaults and expose retry attempts in agent traces."""

    def __init__(self, *, logger, context: dict[str, Any]):
        super().__init__()
        object.__setattr__(self, "_trace_logger", logger)
        object.__setattr__(self, "_trace_context", context)

    def _retryable(self, error: BaseException) -> bool:
        retryable = super()._retryable(error)
        failure_number = self._trace_context.get("failure_number", 0) + 1
        self._trace_context["failure_number"] = failure_number
        trace_event(
            self._trace_logger,
            "agentic_jev_request_retry_check",
            doc_index=self._trace_context["doc_index"],
            total_docs=self._trace_context["total_docs"],
            doc_id=self._trace_context["doc_id"],
            failure_number=failure_number,
            retrying=retryable and failure_number <= self.max_retries,
            error_type=type(error).__name__,
            status_code=getattr(error, "status", None),
            request_id=getattr(error, "request_id", None),
        )
        return retryable


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


def _render_results_for_judge(*, corpus, ranked_doc_ids: list[str], lookup: dict | None) -> str:
    lines = []
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        row, doc_id_int = _judge_row_for_doc_id(corpus=corpus, doc_id=doc_id, lookup=lookup)
        if doc_id_int is None:
            continue
        title = ""
        description = ""
        if row is not None:
            title = str(row.get("title", ""))
            description = str(row.get("description", ""))
        if len(description) > 200:
            description = description[:197] + "..."
        lines.append(f"{idx}. {title} (ID: {doc_id_int})\n{description}")
    return "\n\n".join(lines)


def _judge_row_for_doc_id(*, corpus, doc_id: str, lookup: dict | None):
    try:
        doc_id_int = int(doc_id)
    except (TypeError, ValueError):
        return None, None
    if lookup is not None and doc_id_int in lookup:
        return corpus.iloc[lookup[doc_id_int]], doc_id_int
    if "doc_id" in corpus.columns:
        match = corpus[corpus["doc_id"] == doc_id_int]
        if not match.empty:
            return match.iloc[0], doc_id_int
    return None, doc_id_int


def _judge_is_passing(graded_results: list[GradedSearchResult]) -> bool:
    if not graded_results:
        return False
    return all(item.emoji == "😃" for item in graded_results)


def _oracle_emojis_for_grades(grades: list) -> tuple[dict, list]:
    if not grades:
        raise ValueError("Oracle validator requires at least one grade label.")

    def _grade_key(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return str(value)

    sorted_grades = sorted(grades, key=_grade_key)
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
    emoji_map, grade_ordered = _oracle_emojis_for_grades(grades)
    negative_emoji = emoji_map[grade_ordered[0]]

    grade_by_doc: dict[str, Any] = {}
    if not query_judgments.empty:
        grade_order = {grade: idx for idx, grade in enumerate(grade_ordered)}
        for doc_id, group in query_judgments.groupby("doc_id"):
            best = max(group["grade"], key=lambda value: grade_order.get(value, -1))
            grade_by_doc[str(doc_id)] = best

    graded_results = []
    for doc_id in ranked_doc_ids:
        grade = grade_by_doc.get(str(doc_id))
        emoji = emoji_map.get(grade, negative_emoji)
        title = "Sample"
        row = None
        try:
            doc_id_int = int(doc_id)
        except (TypeError, ValueError):
            doc_id_int = None
        if doc_id_int is not None and lookup is not None and doc_id_int in lookup:
            row = corpus.iloc[lookup[doc_id_int]]
        elif doc_id_int is not None and "doc_id" in corpus.columns:
            match = corpus[corpus["doc_id"] == doc_id_int]
            if not match.empty:
                row = match.iloc[0]
        if row is not None:
            title = str(row.get("title", "Sample"))
        graded_results.append(
            GradedSearchResult(
                emoji=emoji,
                title=title,
                doc_id=str(doc_id),
            )
        )
    return graded_results


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
        process_images=images,
    )
    resp, _, _ = judge_agent.chat(inputs=inputs, agent_state=None, logger=logger)
    parsed = getattr(resp, "output_parsed", None)
    if parsed is None:
        return []
    return list(parsed.graded_results or [])


def _jev_model_name(model: str) -> str:
    provider, separator, model_name = model.partition("/")
    if provider.lower() == "jev":
        return model_name if separator else "jev-latest"
    return model


def _is_valid_jev_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _run_jev_judge(
    *,
    query: str,
    corpus,
    lookup: dict | None,
    ranked_doc_ids: list[str],
    model: str,
    choices: dict[str, str],
    probability_threshold: float,
    confidence_threshold: float,
    judge_prompt: str,
) -> list[dict[str, Any]]:
    if not ranked_doc_ids:
        return []
    client = TypeSafeClient(
        api_key=key_for_provider("typesafe"),
        model=_jev_model_name(model),
    )
    evaluations = []
    for doc_id in ranked_doc_ids:
        result_block = _render_results_for_judge(
            corpus=corpus,
            ranked_doc_ids=[str(doc_id)],
            lookup=lookup,
        )
        row, _ = _judge_row_for_doc_id(corpus=corpus, doc_id=doc_id, lookup=lookup)
        title = str(row.get("title", "")) if row is not None else ""
        if not result_block:
            evaluations.append(
                {
                    "doc_id": str(doc_id),
                    "title": title,
                    "label": None,
                    "probability": None,
                    "confidence": None,
                    "accepted": False,
                }
            )
            continue
        instructions = judge_prompt.format(query=query, results=result_block)
        response = client.system_one(
            state=query,
            questions={
                "relevance": Choice(
                    instructions=instructions,
                    criteria=choices,
                )
            },
        )
        answers = getattr(response, "choices", None)
        answer = answers.get("relevance") if isinstance(answers, dict) else None
        label = getattr(answer, "choice", None)
        confidence = getattr(answer, "confidence", None)
        probabilities = getattr(answer, "probabilities", None)
        probability = (
            probabilities.get(label)
            if isinstance(probabilities, dict) and isinstance(label, str)
            else None
        )
        label_is_valid = isinstance(label, str) and label in choices
        accepted = (
            label_is_valid
            and _is_valid_jev_score(probability)
            and probability > probability_threshold
            and _is_valid_jev_score(confidence)
            and confidence > confidence_threshold
        )
        evaluations.append(
            {
                "doc_id": str(doc_id),
                "title": title,
                "label": label,
                "probability": probability,
                "confidence": confidence,
                "accepted": accepted,
            }
        )
    return evaluations


def _jev_judge_is_passing(evaluations: list[dict[str, Any]]) -> bool:
    if not evaluations:
        return False
    return all(
        evaluation["accepted"] and evaluation["label"] == "Relevant"
        for evaluation in evaluations
    )


def _jev_judge_feedback(evaluations: list[dict[str, Any]]) -> str:
    lines = []
    for idx, evaluation in enumerate(evaluations, start=1):
        label = evaluation["label"]
        if not evaluation["accepted"]:
            label = f"uncertain {label or 'unknown'}"
        probability = evaluation["probability"]
        confidence = evaluation["confidence"]
        probability_text = f"{probability:.3f}" if _is_valid_jev_score(probability) else "n/a"
        confidence_text = f"{confidence:.3f}" if _is_valid_jev_score(confidence) else "n/a"
        title = f" {evaluation['title']}" if evaluation.get("title") else ""
        lines.append(
            f"{idx}. {label} (probability: {probability_text}, "
            f"confidence: {confidence_text}){title} (ID: {evaluation['doc_id']})"
        )
    return "\n".join(lines)


def _run_jev_bag_of_decisions_judge(
    *,
    query: str,
    corpus,
    lookup: dict | None,
    ranked_doc_ids: list[str],
    model: str,
    generator: dict[str, Any],
    positive_probability_threshold: float,
    negative_probability_threshold: float,
    state_format: str,
    agent_state: dict | None,
    logger,
) -> list[dict[str, Any]]:
    """Score each result against an LLM-generated rubric of yes/no questions."""
    if not ranked_doc_ids:
        return []

    rubric_key_payload = json.dumps(
        {"query": query, "generator": generator},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    rubric_key = hashlib.sha256(rubric_key_payload).hexdigest()
    rubric_cache = None
    if agent_state is not None:
        rubric_cache = agent_state.setdefault(
            "jev_bag_of_decisions_judge_rubrics", {}
        )
        if not isinstance(rubric_cache, dict):
            rubric_cache = {}
            agent_state["jev_bag_of_decisions_judge_rubrics"] = rubric_cache

    if rubric_cache is not None and rubric_key in rubric_cache:
        raw_questions = rubric_cache[rubric_key]
        decisions = [
            DecisionQuestion(
                instructions=item["instructions"],
                criteria=item.get("criteria"),
            )
            for item in raw_questions
            if isinstance(item, dict)
            and isinstance(item.get("instructions"), str)
        ]
        trace_event(
            logger,
            "agentic_jev_rubric_reused",
            query=query,
            question_count=len(decisions),
        )
    else:
        generator_options = {
            key: generator[key]
            for key in (
                "model",
                "system_prompt",
                "prompt",
                "reasoning",
                "temperature",
                "verbosity",
                "no_cache",
            )
            if key in generator
        }
        rubric_started = monotonic()
        set_trace_stage(
            agent_state,
            "jev_rubric_generation",
            generator_model=generator_options.get("model"),
        )
        trace_event(
            logger,
            "agentic_jev_rubric_start",
            generator_model=generator_options.get("model"),
        )
        try:
            question_generator = DecisionGenerator(**generator_options)
            decisions = question_generator.generate(query)
        except Exception as exc:
            trace_event(
                logger,
                "agentic_jev_rubric_error",
                elapsed_seconds=round(monotonic() - rubric_started, 3),
                error_type=type(exc).__name__,
            )
            raise
        decisions = [
            decision
            for decision in decisions
            if isinstance(decision, DecisionQuestion)
            and isinstance(decision.instructions, str)
            and decision.instructions.strip()
        ]
        if rubric_cache is not None:
            rubric_cache[rubric_key] = [
                {
                    "instructions": decision.instructions,
                    "criteria": (
                        dict(decision.criteria)
                        if decision.criteria is not None
                        else None
                    ),
                }
                for decision in decisions
            ]
        trace_event(
            logger,
            "agentic_jev_rubric_complete",
            elapsed_seconds=round(monotonic() - rubric_started, 3),
            question_count=len(decisions),
        )

    if not decisions:
        return [
            {
                "doc_id": str(doc_id),
                "title": "",
                "score": None,
                "positive_decisions": [],
                "negative_decisions": [],
            }
            for doc_id in ranked_doc_ids
        ]

    client = TypeSafeClient(
        api_key=key_for_provider("typesafe"),
        model=_jev_model_name(model),
    )
    evaluations: list[dict[str, Any]] = []
    total_docs = len(ranked_doc_ids)
    for doc_index, doc_id in enumerate(ranked_doc_ids, start=1):
        document_started = monotonic()
        set_trace_stage(
            agent_state,
            "jev_document_evaluation",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(doc_id),
            question_count=len(decisions),
        )
        trace_event(
            logger,
            "agentic_jev_document_start",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(doc_id),
            question_count=len(decisions),
        )
        row, normalized_doc_id = _judge_row_for_doc_id(
            corpus=corpus,
            doc_id=doc_id,
            lookup=lookup,
        )
        title = str(row.get("title", "")) if row is not None else ""
        if row is None or normalized_doc_id is None:
            trace_event(
                logger,
                "agentic_jev_document_complete",
                doc_index=doc_index,
                total_docs=total_docs,
                doc_id=str(doc_id),
                elapsed_seconds=round(monotonic() - document_started, 3),
                outcome="document_not_found",
            )
            evaluations.append(
                {
                    "doc_id": str(doc_id),
                    "title": title,
                    "score": None,
                    "positive_decisions": [],
                    "negative_decisions": [],
                }
            )
            continue

        row_values = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        try:
            state = state_format.format_map({**row_values, "query": query})
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError(
                "Could not render params.state_format for a corpus row."
            ) from exc

        questions = {}
        for index, decision in enumerate(decisions):
            if decision.criteria is None:
                question = Noul(instructions=decision.instructions)
            else:
                question = Noul(
                    instructions=decision.instructions,
                    criteria=NoulCriteria(**decision.criteria),
                )
            questions[f"decision_{index}"] = question
        request_started = monotonic()
        trace_event(
            logger,
            "agentic_jev_request_start",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(normalized_doc_id),
            question_count=len(questions),
        )
        retry_policy = _TracingRetryPolicy(
            logger=logger,
            context={
                "doc_index": doc_index,
                "total_docs": total_docs,
                "doc_id": str(normalized_doc_id),
            },
        )
        try:
            response = client.system_one(
                state=state,
                questions=questions,
                retry=retry_policy,
            )
        except Exception as exc:
            trace_event(
                logger,
                "agentic_jev_request_error",
                doc_index=doc_index,
                total_docs=total_docs,
                doc_id=str(normalized_doc_id),
                elapsed_seconds=round(monotonic() - request_started, 3),
                error_type=type(exc).__name__,
            )
            raise
        answers = getattr(response, "answers", None)
        answers = answers if isinstance(answers, dict) else {}

        score = 0.0
        valid_answers = 0
        positive_decisions = []
        negative_decisions = []
        for index, decision in enumerate(decisions):
            answer = answers.get(f"decision_{index}")
            probability = getattr(answer, "noul", None)
            if not _is_valid_jev_score(probability):
                continue
            probability = float(probability)
            score += probability
            valid_answers += 1
            item = {
                "question": decision.instructions,
                "probability": probability,
            }
            if probability >= positive_probability_threshold:
                positive_decisions.append(item)
            elif probability <= negative_probability_threshold:
                negative_decisions.append(item)

        evaluations.append(
            {
                "doc_id": str(normalized_doc_id),
                "title": title,
                "score": score if valid_answers else None,
                "positive_decisions": positive_decisions,
                "negative_decisions": negative_decisions,
            }
        )
        trace_event(
            logger,
            "agentic_jev_request_complete",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(normalized_doc_id),
            elapsed_seconds=round(monotonic() - request_started, 3),
            request_id=getattr(response, "request_id", None),
            valid_answers=valid_answers,
            score=score if valid_answers else None,
        )
        trace_event(
            logger,
            "agentic_jev_document_complete",
            doc_index=doc_index,
            total_docs=total_docs,
            doc_id=str(normalized_doc_id),
            elapsed_seconds=round(monotonic() - document_started, 3),
            outcome="evaluated",
        )
    return evaluations


def _jev_bag_of_decisions_judge_is_passing(
    evaluations: list[dict[str, Any]],
) -> bool:
    if not evaluations:
        return False
    return all(
        evaluation.get("score") is not None
        and bool(evaluation.get("positive_decisions"))
        and not evaluation.get("negative_decisions")
        for evaluation in evaluations
    )


def _jev_bag_of_decisions_judge_feedback(
    evaluations: list[dict[str, Any]],
) -> str:
    lines = []
    for index, evaluation in enumerate(evaluations, start=1):
        title = f" {evaluation['title']}" if evaluation.get("title") else ""
        score = evaluation.get("score")
        score_text = (
            f"{score:.3f}"
            if isinstance(score, (int, float))
            and not isinstance(score, bool)
            and math.isfinite(score)
            and score >= 0
            else "n/a"
        )
        lines.append(
            f"{index}.{title} (ID: {evaluation['doc_id']}; "
            f"rubric score: {score_text})"
        )
        positive_decisions = evaluation.get("positive_decisions") or []
        negative_decisions = evaluation.get("negative_decisions") or []
        for decision in positive_decisions:
            lines.append(
                f"   👍 {decision['question']} "
                f"(P(yes)={decision['probability']:.3f})"
            )
        for decision in negative_decisions:
            lines.append(
                f"   👎 {decision['question']} "
                f"(P(yes)={decision['probability']:.3f})"
            )
        if not positive_decisions and not negative_decisions:
            lines.append("   No criteria cleared either probability threshold.")
    if not lines:
        return "No results were available to evaluate."
    return "\n".join(lines)
