from __future__ import annotations

from textwrap import dedent


DEFAULT_SYSTEM_PROMPT = dedent(
    """
    Your task is to improve the reranker code so it returns more relevant results.

    Edit the reranker python module using apply_patch.

    You can run the reranker using run_reranker, which takes a query and returns ranked matches.
    You can evaluate the reranker using run_evals, which returns NDCG scores and mean NDCG.

    If NDCG does not improve after edits, revert changes using revert_changes.

    Your code must include a function named {rerank_name}. It takes a search tool and a query
    string and returns a list of document ids ordered most relevant to least.

    Keep a wrapper function named reranker that calls {rerank_name}.
    """
).strip()


def build_system_prompt(
    base_prompt: str | None,
    *,
    dataset: str,
    rerank_name: str,
    search_tool_names: list[str],
    code: str,
) -> str:
    prompt = base_prompt or DEFAULT_SYSTEM_PROMPT.format(rerank_name=rerank_name)
    tool_block = "\n".join(f"- {name}" for name in search_tool_names)
    return dedent(
        f"""
        {prompt}

        Dataset: {dataset}
        Available search tools:
        {tool_block}

        Reranker code to improve:

        {code}
        """
    ).strip()
