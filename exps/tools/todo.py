from __future__ import annotations

def _todos_list(agent_state: dict | None) -> list[dict]:
    if agent_state is None:
        return []
    todos = agent_state.get("todos")
    if todos is None:
        todos = []
        agent_state["todos"] = todos
    return todos


def make_todo_write_tool(*_args, **_kwargs):
    def todo_write(todo: str, status: str, agent_state=None) -> str:
        """Write a todo entry to the in-memory todo list."""
        print(f"TODO: {todo} (status: {status})")
        todos = _todos_list(agent_state)
        todos.append({"todo": todo, "status": status})
        return "OK"

    return todo_write


def make_todo_read_tool(*_args, **_kwargs):
    def todo_read(agent_state=None) -> str:
        """Read the in-memory todo list and return its contents."""
        todos = _todos_list(agent_state)
        if not todos:
            return ""
        lines = [f"{item.get('status', '')}\t{item.get('todo', '')}" for item in todos]
        return "\n".join(lines) + "\n"

    return todo_read
