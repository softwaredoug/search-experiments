from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from cheat_at_search.enrich.enrich import AutoEnricher


def make_query_rewrite_tool(
    corpus,
    *,
    tool_config: dict[str, Any] | None = None,
    **_unused,
):
    config = tool_config or {}
    model = config.get("model") or "openai/gpt-5-mini"
    max_alternatives = config.get("max_alternatives") or 5
    temperature = config.get("temperature")
    reasoning_effort = config.get("reasoning_effort")
    verbosity = config.get("verbosity")

    if "/" not in model:
        model = f"openai/{model}"

    class QueryRewriteResponse(BaseModel):
        original_query: str = Field(description="The original query string")
        rewriters: list[str] = Field(description="Alternate query forms including the original query")

    system_prompt = (
        "You rewrite search queries to alternate forms without changing meaning. "
        "Only apply spelling corrections, acronym expansions, or acronym forms of the same terms. "
        "Do not add synonyms or new concepts. Return a short list of alternates."
    )
    enricher = AutoEnricher(
        model=model,
        system_prompt=system_prompt,
        response_model=QueryRewriteResponse,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        verbosity=verbosity,
    )

    def query_rewrite(
        query: str,
        max_alternatives: int = max_alternatives,
        agent_state=None,
    ) -> dict[str, Any]:
        """Rewrite a query into alternate forms for lexical search."""
        if not isinstance(query, str) or not query.strip():
            original = "" if isinstance(query, str) else str(query)
            return {"original_query": original, "rewriters": [original]}
        prompt = (
            "Rewrite the query with alternate spellings or acronym expansions only. "
            "Do not add synonyms. Ensure the original query appears in the list. "
            "Return JSON with keys 'original_query' and 'rewriters'.\n\n"
            f"Query: {query}\n"
            f"Max alternates: {max_alternatives}"
        )
        parsed = enricher.enrich(prompt)
        candidates = parsed.rewriters if parsed and parsed.rewriters else []
        deduped = []
        seen = set()
        for item in candidates:
            if not isinstance(item, str):
                continue
            normalized = item.strip()
            if not normalized or normalized in seen:
                continue
            deduped.append(normalized)
            seen.add(normalized)
        original = query.strip()
        if original:
            if original in seen:
                deduped.remove(original)
            deduped.insert(0, original)
        rewriters = deduped[: max(1, int(max_alternatives))]
        return {"original_query": original, "rewriters": rewriters}

    query_rewrite.__name__ = "query_rewrite"
    query_rewrite.__doc__ = (
        "Rewrite a query into alternate forms (spelling/acronym variants only)."
    )
    return query_rewrite
