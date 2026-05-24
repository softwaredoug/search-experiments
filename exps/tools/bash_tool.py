from __future__ import annotations

from exps.tools.bash_service import start_bash_service
from exps.tools.filesystem_index import ensure_filesystem_on_disk


_MAX_OUTPUT_CHARS = 8000


def _truncate_output(text: str, *, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[output truncated]"


def _make_bash_tool(corpus, *, dataset_name: str | None, variant: str):
    if not dataset_name:
        raise ValueError("bash tool requires dataset_name")
    dataset_dir = ensure_filesystem_on_disk(corpus, dataset_name=dataset_name, variant=variant)
    service = start_bash_service(dataset_dir)

    def bash(command: str, timeout: int = 30, agent_state=None) -> str:
        """Execute a bash command inside the sandboxed filesystem service.

        Commands run inside /corpus which maps to the dataset directory. There may be
        subdirectories for any category / subcatgory organization.

        Filenames are document title slug with id txt, ie "red-shoes-1234.txt".
        Document body is:

        ```
        <Title> (ID: <ID>)

        <Description + Other Metadata>
        ```


        ```
        # Red Shoes (ID: 1234)

        These are the best red shoes you'll ever find. They're super comfy and stylish.
        """
        print(command)
        if agent_state is not None:
            logger = agent_state.get("trace_logger")
            if logger is not None:
                logger.info("bash_command %s", command)
        output = service.execute(command, timeout=timeout)
        return _truncate_output(output)

    bash.__name__ = "bash" if variant == "default" else f"bash_{variant}"
    bash.__doc__ = (
        "Execute bash commands inside the sandboxed filesystem service. "
        f"Search within {dataset_dir}. Commands run in /corpus."
    )
    return bash


def make_bash_tool(
    corpus,
    *,
    dataset_name: str | None = None,
    tool_config: dict | None = None,
    **_unused,
):
    return _make_bash_tool(corpus, dataset_name=dataset_name, variant="default")


def make_bash_wands_tool(
    corpus,
    *,
    dataset_name: str | None = None,
    tool_config: dict | None = None,
    **_unused,
):
    return _make_bash_tool(corpus, dataset_name=dataset_name, variant="wands")
