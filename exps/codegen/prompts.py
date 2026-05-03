from __future__ import annotations

from textwrap import dedent


DEFAULT_SYSTEM_PROMPT = dedent(
    """
    Your task is to improve the reranker code so it returns more relevant results.

    Edit the reranker python module using apply_patch.

    You can run the reranker using run_reranker, which takes a query and returns ranked matches.
    You can evaluate the reranker using run_evals, which returns NDCG scores and mean NDCG.

    If NDCG does not improve after edits, revert changes using revert_changes.

    Your code must include a function named {rerank_name}. It takes the query first, followed by
    the available search tools, plus **kwargs for hidden runtime args, and returns a list of
    document ids ordered most relevant to least.

    """
).strip()


def build_system_prompt(
    base_prompt: str | None,
    *,
    dataset: str,
    rerank_name: str,
    search_tool_names: list[str],
    search_tool_docs: list[str],
    raw_tool_names: list[str] | None = None,
    raw_tool_docs: list[str] | None = None,
    rerank_params: list[str],
    code: str,
) -> str:
    prompt = base_prompt or DEFAULT_SYSTEM_PROMPT.format(rerank_name=rerank_name)
    tool_lines = []
    for name, doc in zip(search_tool_names, search_tool_docs):
        doc_str = (doc or "").strip()
        if doc_str:
            tool_lines.append(f"- {name}: {doc_str}")
        else:
            tool_lines.append(f"- {name}")
    tool_block = "\n".join(tool_lines)
    raw_lines = []
    if raw_tool_names:
        for name, doc in zip(raw_tool_names, raw_tool_docs or []):
            doc_str = (doc or "").strip()
            if doc_str:
                raw_lines.append(f"### {name}\n\n{doc_str}")
            else:
                raw_lines.append(f"### {name}")
    raw_block = "\n\n".join(raw_lines)
    params_block = ", ".join(rerank_params)
    return dedent(
        f"""
        {prompt}

        Dataset: {dataset}
        Available search tools:
        {tool_block}

        Reranker signature parameters:
        {params_block}

        Reranker code to improve:

        {code}

        {"## Additionally injected search code\n\nThe following functions are available to generated code, injected into the rerank function:\n\n" + raw_block if raw_block else ""}
        """
    ).strip()
