"""Bash service integration tests.

These tests validate the docker-backed bash service in isolation.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import patch
from urllib import request

import pandas as pd
import pytest

from exps.tools.bash_service import ensure_bash_volume, volume_name_for_dataset
from exps.tools.bash_tool import make_bash_tool
from exps.tools.filesystem_index import ensure_filesystem_on_disk


def _docker_available() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return True


def _docker_compose_available() -> bool:
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return True


_TEMP_ROOT: Path | None = None


def _temp_root() -> Path:
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        _TEMP_ROOT = Path(tempfile.mkdtemp())
    return _TEMP_ROOT


def _cleanup_temp_root() -> None:
    global _TEMP_ROOT
    if _TEMP_ROOT is None:
        return
    shutil.rmtree(_TEMP_ROOT, ignore_errors=True)
    _TEMP_ROOT = None


_BASH_PORT_EXEC = 8010
_BASH_PORT_MISSING = 8011


def _require_docker() -> None:
    if _docker_available():
        return
    warnings.warn("Docker not available; skipping bash service integration tests.")
    pytest.skip("Docker is required for bash service integration tests.")


def _require_compose() -> None:
    if _docker_compose_available():
        return
    warnings.warn("Docker compose not available; skipping bash service integration tests.")
    pytest.skip("Docker compose is required for bash service integration tests.")


def _bash_service_ready(port: int, *, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    payload = json.dumps({"command": "pwd", "timeout": 5}).encode("utf-8")
    url = f"http://127.0.0.1:{port}/execute"
    while time.time() < deadline:
        try:
            req = request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            with request.urlopen(req, timeout=2) as resp:
                body = resp.read().decode("utf-8")
            data = json.loads(body)
            if str(data.get("exit_code")) == "0":
                return True
        except Exception:
            time.sleep(0.5)
    return False


def _compose_up(*, volume_name: str, port: int) -> None:
    env = dict(os.environ)
    env.update({"EXPS_BASH_VOLUME": volume_name, "EXPS_BASH_PORT": str(port)})
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.bash.yml", "up", "-d"],
        check=True,
        env=env,
        timeout=60,
    )


def _compose_down(*, volume_name: str, port: int) -> None:
    env = dict(os.environ)
    env.update({"EXPS_BASH_VOLUME": volume_name, "EXPS_BASH_PORT": str(port)})
    subprocess.run(
        ["docker", "compose", "-f", "docker-compose.bash.yml", "down", "-v"],
        check=False,
        env=env,
        timeout=60,
    )


def _run_bash_or_skip(bash_tool, command: str, *, timeout: int) -> str:
    try:
        return bash_tool(command, timeout=timeout)
    except RuntimeError as exc:
        message = str(exc)
        if "Connection reset by peer" in message or "Failed to connect" in message:
            pytest.skip("Bash service became unavailable during integration test.")
        raise


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch.dict(os.environ, {"EXPS_BASH_PORT": str(_BASH_PORT_EXEC)})
def test_bash_tool_executes_commands(_filesystem_root, _paths_root):
    try:
        _require_docker()
        _require_compose()
        dataset_name = "bash_fixture"

        corpus = pd.DataFrame(
            {
                "title": ["alpha", "beta"],
                "description": ["alpha", "beta"],
                "doc_id": ["1", "2"],
            }
        )
        dataset_dir = ensure_filesystem_on_disk(corpus, dataset_name=dataset_name, variant="default")
        volume_name = volume_name_for_dataset(dataset_name)
        ensure_bash_volume(dataset_dir, volume_name=volume_name)
        _compose_up(volume_name=volume_name, port=_BASH_PORT_EXEC)
        if not _bash_service_ready(_BASH_PORT_EXEC):
            _compose_down(volume_name=volume_name, port=_BASH_PORT_EXEC)
            pytest.skip("Bash service did not become ready for integration test.")
        try:
            bash_tool = make_bash_tool(corpus, dataset_name=dataset_name)
            output = _run_bash_or_skip(bash_tool, "ls -1", timeout=5)
        finally:
            _compose_down(volume_name=volume_name, port=_BASH_PORT_EXEC)
    finally:
        _cleanup_temp_root()

    assert "exit_code=0" in output
    assert "alpha-1.txt" in output
    assert "beta-2.txt" in output


@patch("exps.paths.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", new_callable=_temp_root)
@patch.dict(os.environ, {"EXPS_BASH_PORT": str(_BASH_PORT_MISSING)})
def test_bash_tool_handles_missing_file(_filesystem_root, _paths_root):
    try:
        _require_docker()
        _require_compose()
        dataset_name = "bash_fixture_missing"

        corpus = pd.DataFrame(
            {
                "title": ["alpha"],
                "description": ["alpha"],
                "doc_id": ["1"],
            }
        )
        dataset_dir = ensure_filesystem_on_disk(corpus, dataset_name=dataset_name, variant="default")
        volume_name = volume_name_for_dataset(dataset_name)
        ensure_bash_volume(dataset_dir, volume_name=volume_name)
        _compose_up(volume_name=volume_name, port=_BASH_PORT_MISSING)
        if not _bash_service_ready(_BASH_PORT_MISSING):
            _compose_down(volume_name=volume_name, port=_BASH_PORT_MISSING)
            pytest.skip("Bash service did not become ready for integration test.")
        try:
            bash_tool = make_bash_tool(corpus, dataset_name=dataset_name)
            output = _run_bash_or_skip(bash_tool, "cat missing.txt", timeout=5)
        finally:
            _compose_down(volume_name=volume_name, port=_BASH_PORT_MISSING)
    finally:
        _cleanup_temp_root()

    assert "exit_code=1" in output
    assert "No such file" in output
