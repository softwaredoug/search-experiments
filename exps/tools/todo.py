from __future__ import annotations

from pathlib import Path


def _todos_path(agent_state: dict | None) -> Path | None:
    if not agent_state:
        return None
    run_dir = agent_state.get("run_dir")
    if not run_dir:
        return None
    return Path(run_dir) / "todos.txt"


def make_todowrite_tool(*_args, **_kwargs):
    def todowrite(todo: str, status: str, agent_state=None) -> str:
        """Write a todo entry to todos.txt in the run folder."""
        path = _todos_path(agent_state)
        if path is None:
            return "Error! run_dir not set in agent_state."
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = f"{status}\t{todo}".strip() + "\n"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        return "OK"

    return todowrite


def make_todoread_tool(*_args, **_kwargs):
    def todoread(agent_state=None) -> str:
        """Read todos.txt from the run folder and return its contents."""
        path = _todos_path(agent_state)
        if path is None:
            return "Error! run_dir not set in agent_state."
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    return todoread
