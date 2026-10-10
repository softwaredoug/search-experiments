"""Types, constants, and corpus-result utilities shared by judge conditions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AllowedEmoji = Literal["🤩", "😃", "😐", "😞"]
PASSING_EMOJI = "😃"
MAX_JUDGE_DESCRIPTION_CHARS = 200


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


def _render_results_for_judge(
    *, corpus, ranked_doc_ids: list[str], lookup: dict | None
) -> str:
    lines = []
    for idx, doc_id in enumerate(ranked_doc_ids, start=1):
        row, doc_id_int = _judge_row_for_doc_id(
            corpus=corpus, doc_id=doc_id, lookup=lookup
        )
        if doc_id_int is None:
            continue
        title = str(row.get("title", "")) if row is not None else ""
        description = str(row.get("description", "")) if row is not None else ""
        if len(description) > MAX_JUDGE_DESCRIPTION_CHARS:
            description = description[: MAX_JUDGE_DESCRIPTION_CHARS - 3] + "..."
        lines.append(f"{idx}. {title} (ID: {doc_id_int})\n{description}")
    return "\n\n".join(lines)
