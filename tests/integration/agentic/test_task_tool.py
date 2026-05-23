import os

from exps.agentic.strategy import DEFAULT_SYSTEM_PROMPT, SearchResultsIds
from exps.agentic.task import build_task_tool
from exps.datasets import get_dataset
from exps.tools import build_search_tools


def test_task_tool_returns_search_results():
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for agentic tests.")

    dataset = get_dataset("doug_blog")
    tools = build_search_tools(
        dataset.corpus,
        ["bm25"],
        embeddings_device=None,
        dataset_name="doug_blog",
    )
    task_tool = build_task_tool(
        search_tools=tools,
        model="gpt-5-mini",
        reasoning="low",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_model=SearchResultsIds,
    )

    results = task_tool("salon chair", top_k=5, agent_state={})

    assert isinstance(results, list)
    assert results
    assert isinstance(results[0], dict)
    assert "doc_id" in results[0] or "id" in results[0]
