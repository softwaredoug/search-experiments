from __future__ import annotations

import http.client
import os
import threading
from urllib import error as urllib_error

from exps.tools.bash_service import (
    BashService,
    bash_service_running,
    ensure_bash_volume,
    volume_name_for_dataset,
)
from exps.tools.filesystem_index import ensure_filesystem_on_disk, filesystem_root


_MAX_OUTPUT_CHARS = 8000
_DEFAULT_MAX_TIMEOUT = 30
_SERVICE_CACHE: dict[int, BashService] = {}
_SERVICE_LOCKS: dict[int, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def _truncate_output(text: str, *, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[output truncated]"


def _parse_exit_code(output: str) -> int | None:
    if not output.startswith("exit_code="):
        return None
    first_line = output.splitlines()[0]
    try:
        return int(first_line.split("=", 1)[1])
    except (IndexError, ValueError):
        return None


def _get_service(port: int) -> tuple[BashService, threading.Lock]:
    with _CACHE_LOCK:
        service = _SERVICE_CACHE.get(port)
        if service is None:
            service = BashService(port)
            _SERVICE_CACHE[port] = service
        lock = _SERVICE_LOCKS.setdefault(port, threading.Lock())
    return service, lock


def _compose_instructions(*, volume_name: str, port: int) -> str:
    return (
        "Start the bash service with:\n"
        f"EXPS_BASH_VOLUME={volume_name} EXPS_BASH_PORT={port} \\n"
        "docker compose -f docker-compose.bash.yml up -d"
    )


def _make_bash_tool(corpus, *, dataset_name: str | None, variant: str):
    if not dataset_name:
        raise ValueError("bash tool requires dataset_name")
    port = int(os.getenv("EXPS_BASH_PORT", "8000"))
    dataset_root = filesystem_root(dataset_name)
    filesystem_created = not dataset_root.exists()
    dataset_dir = ensure_filesystem_on_disk(corpus, dataset_name=dataset_name, variant=variant)
    volume_name = volume_name_for_dataset(dataset_name)
    volume_created = ensure_bash_volume(dataset_dir, volume_name=volume_name)
    if filesystem_created or volume_created:
        raise RuntimeError(
            "Bash filesystem or volume created; rerun after starting docker compose. "
            + _compose_instructions(volume_name=volume_name, port=port)
        )
    if not bash_service_running(port):
        raise RuntimeError(
            "Bash service is not running. "
            + _compose_instructions(volume_name=volume_name, port=port)
        )
    service, service_lock = _get_service(port)

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
        max_timeout = int(os.getenv("EXPS_BASH_MAX_TIMEOUT", str(_DEFAULT_MAX_TIMEOUT)))
        effective_timeout = min(timeout, max_timeout)
        with service_lock:
            try:
                output = service.execute(command, timeout=effective_timeout)
                exit_code = _parse_exit_code(output)
                if exit_code == 124:
                    raise RuntimeError(
                        f"Bash command timed out after {effective_timeout}s."
                    )
                return _truncate_output(output)
            except (TimeoutError, OSError, http.client.RemoteDisconnected, urllib_error.URLError) as exc:
                raise RuntimeError(f"Bash command failed: {exc}")

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
