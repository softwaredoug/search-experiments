from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from typing import Any

_QUERY_ARGUMENTS = ("query", "keywords", "question", "product_description")
_MAX_TOP_K = 100


def _query_argument(tool: Callable) -> str:
    signature = inspect.signature(tool)
    query_name = next(
        (name for name in _QUERY_ARGUMENTS if name in signature.parameters), None
    )
    if query_name is None:
        raise ValueError(
            f"Composite search tool {tool.__name__} must accept one of "
            f"{', '.join(_QUERY_ARGUMENTS)}."
        )

    unsupported_required = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.name not in {query_name, "top_k", "agent_state"}
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    ]
    if unsupported_required:
        names = ", ".join(unsupported_required)
        raise ValueError(
            f"Composite search tool {tool.__name__} has unsupported required "
            f"arguments: {names}."
        )
    return query_name


def make_rrf_tool(
    search_tools: list[Callable],
    weights: list[float] | None = None,
    *,
    rank_constant: int = 60,
):
    """Build a weighted reciprocal-rank-fusion search tool.

    The configured search tools and RRF parameters are injected by the tool
    builder. Every child must accept a query string and return ranked document
    dictionaries containing an ``id`` (or ``doc_id``).
    """
    if not isinstance(search_tools, list) or not search_tools:
        raise ValueError("RRF requires at least one search tool.")

    if weights is None:
        weights = [1.0] * len(search_tools)
    if not isinstance(weights, list) or len(weights) != len(search_tools):
        raise ValueError("RRF weights must contain one value per search tool.")

    parsed_weights = []
    for weight in weights:
        if isinstance(weight, bool):
            raise ValueError("RRF weights must be finite, non-negative numbers.")
        try:
            parsed_weight = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError("RRF weights must be finite, non-negative numbers.") from exc
        if not math.isfinite(parsed_weight) or parsed_weight < 0:
            raise ValueError("RRF weights must be finite, non-negative numbers.")
        parsed_weights.append(parsed_weight)

    if isinstance(rank_constant, bool) or not isinstance(rank_constant, int) or rank_constant < 0:
        raise ValueError("RRF rank_constant must be a non-negative integer.")

    child_arguments = []
    for tool in search_tools:
        if not callable(tool):
            raise ValueError("RRF search_tools entries must be callable.")
        child_arguments.append(_query_argument(tool))

    def search_composite(
        query: str,
        top_k: int = 5,
        agent_state=None,
    ) -> list[dict]:
        """Search configured tools and fuse their ranked results with RRF."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 0:
            raise ValueError("top_k must be a non-negative integer.")
        if top_k > _MAX_TOP_K:
            return "Error! top_k must be <= 100."
        if top_k == 0:
            return []

        fused_by_id: dict[str, dict[str, Any]] = {}
        for tool, query_name, weight in zip(search_tools, child_arguments, parsed_weights):
            child_kwargs = {query_name: query, "top_k": top_k, "agent_state": agent_state}
            signature = inspect.signature(tool)
            child_kwargs = {
                name: value
                for name, value in child_kwargs.items()
                if name in signature.parameters
            }
            results = tool(**child_kwargs)
            if isinstance(results, str):
                return results
            if not isinstance(results, list):
                raise ValueError(
                    f"Composite search tool {tool.__name__} must return a list of results."
                )

            for rank, result in enumerate(results, start=1):
                if not isinstance(result, dict):
                    raise ValueError(
                        f"Composite search tool {tool.__name__} returned a non-mapping result."
                    )
                doc_id = result.get("id", result.get("doc_id"))
                if doc_id is None:
                    raise ValueError(
                        f"Composite search tool {tool.__name__} returned a result without an id."
                    )
                key = str(doc_id)
                fused = fused_by_id.get(key)
                if fused is None:
                    fused = dict(result)
                    fused["id"] = doc_id
                    fused["score"] = 0.0
                    fused_by_id[key] = fused
                fused["score"] += weight / (rank_constant + rank)

        ranked_results = sorted(
            fused_by_id.values(),
            key=lambda result: result["score"],
            reverse=True,
        )
        return ranked_results[:top_k]

    return search_composite
