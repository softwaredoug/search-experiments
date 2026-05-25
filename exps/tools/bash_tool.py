from __future__ import annotations

import http.client
import threading
from pathlib import Path
from urllib import error as urllib_error

from exps.tools.bash_service import BashService, start_bash_service
from exps.tools.filesystem_index import ensure_filesystem_on_disk


_MAX_OUTPUT_CHARS = 8000
_SERVICE_CACHE: dict[tuple[str, str], BashService] = {}
_SERVICE_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def _truncate_output(text: str, *, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[output truncated]"


def _get_service(dataset_dir: str, variant: str) -> tuple[BashService, threading.Lock]:
    key = (dataset_dir, variant)
    with _CACHE_LOCK:
        service = _SERVICE_CACHE.get(key)
        if service is None or service.port is None:
            service = start_bash_service(Path(dataset_dir))
            _SERVICE_CACHE[key] = service
        lock = _SERVICE_LOCKS.setdefault(key, threading.Lock())
    return service, lock


def _make_bash_tool(corpus, *, dataset_name: str | None, variant: str):
    if not dataset_name:
        raise ValueError("bash tool requires dataset_name")
    dataset_dir = ensure_filesystem_on_disk(corpus, dataset_name=dataset_name, variant=variant)
    service, service_lock = _get_service(str(dataset_dir), variant)

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
        if agent_state is not None:
            logger = agent_state.get("trace_logger")
            if logger is not None:
                logger.info("bash_command %s", command)
        nonlocal service
        with service_lock:
            try:
                output = service.execute(command, timeout=timeout)
                return _truncate_output(output)
            except (TimeoutError, OSError, http.client.RemoteDisconnected, urllib_error.URLError) as exc:
                if agent_state is not None:
                    logger = agent_state.get("trace_logger")
                    if logger is not None:
                        logger.info("bash_service_restart %s", str(exc))
                try:
                    service.stop()
                except Exception:
                    pass
                service = start_bash_service(dataset_dir)
                _SERVICE_CACHE[(str(dataset_dir), variant)] = service
                try:
                    output = service.execute(command, timeout=timeout)
                    return _truncate_output(output)
                except (TimeoutError, OSError, http.client.RemoteDisconnected, urllib_error.URLError) as retry_exc:
                    return f"Error! bash command failed: {retry_exc}"

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
