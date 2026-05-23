from __future__ import annotations

import json
from typing import Any, Callable
from cheat_at_search.agent.openai_agent import OpenAIAgent


def _append_tool_output(results: list[dict], output: Any) -> None:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            return
    if isinstance(output, list):
        for item in output:
            if isinstance(item, dict):
                results.append(item)
        return
    if isinstance(output, dict):
        results.append(output)


def _log_subagent_outputs(agent_state: dict, items: list[dict]) -> None:
    logger = agent_state.get("trace_logger")
    if logger is None:
        return
    for item in items:
        logger.info("subagent_output %s", item)


def _log_task_tool_call(agent_state: dict, task: str) -> None:
    logger = agent_state.get("trace_logger")
    if logger is None:
        return
    logger.info("task_tool_call %s", {"task": task})


def _log_task_tool_result(agent_state: dict, count: int) -> None:
    logger = agent_state.get("trace_logger")
    if logger is None:
        return
    logger.info("task_tool_result %s", {"result_count": count})


def build_task_tool(
    *,
    search_tools: list[callable],
    model: str,
    reasoning: str,
    system_prompt: str,
) -> Callable[[str, int, dict | None], list[dict]]:
    """Build a task tool for orchestrated agent topologies."""

    def task_tool(task: str, agent_state: dict | None = None) -> list[dict]:
        """Delegate a search task to a subagent and return tool results."""
        if agent_state is None:
            agent_state = {}
        _log_task_tool_call(agent_state, task)
        agent = OpenAIAgent(
            tools=search_tools,
            model=f"openai/{model}" if "/" not in model else model,
            reasoning_level=reasoning,
            response_model=None,
        )
        inputs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {task}\n"},
        ]
        previous_inputs = list(inputs)
        logger = agent_state.get("trace_logger")
        resp, inputs, _ = agent.chat(inputs=inputs, agent_state=agent_state, logger=logger)
        output = resp.output[-1].content[-1].text
        return output

    return task_tool
