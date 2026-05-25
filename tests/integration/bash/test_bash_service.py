"""Bash service integration tests.

These tests validate the docker-backed bash service in isolation.
"""

import subprocess
import warnings

import pandas as pd
import pytest

from exps.tools.bash_tool import make_bash_tool


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


def _require_docker() -> None:
    if _docker_available():
        return
    warnings.warn("Docker not available; skipping bash service integration tests.")
    pytest.skip("Docker is required for bash service integration tests.")


def test_bash_tool_executes_commands(tmp_path, monkeypatch):
    _require_docker()

    trace_root = tmp_path / "search-experiments"
    monkeypatch.setattr("exps.paths.SEARCH_EXPERIMENTS_ROOT", trace_root)
    monkeypatch.setattr("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", trace_root)

    corpus = pd.DataFrame(
        {
            "title": ["alpha", "beta"],
            "description": ["alpha", "beta"],
            "doc_id": ["1", "2"],
        }
    )
    bash_tool = make_bash_tool(corpus, dataset_name="bash_fixture")
    output = bash_tool("ls -1", timeout=5)

    assert "exit_code=0" in output
    assert "alpha-1.txt" in output
    assert "beta-2.txt" in output


def test_bash_tool_handles_missing_file(tmp_path, monkeypatch):
    _require_docker()

    trace_root = tmp_path / "search-experiments"
    monkeypatch.setattr("exps.paths.SEARCH_EXPERIMENTS_ROOT", trace_root)
    monkeypatch.setattr("exps.tools.filesystem_index.SEARCH_EXPERIMENTS_ROOT", trace_root)

    corpus = pd.DataFrame(
        {
            "title": ["alpha"],
            "description": ["alpha"],
            "doc_id": ["1"],
        }
    )
    bash_tool = make_bash_tool(corpus, dataset_name="bash_fixture_missing")
    output = bash_tool("cat missing.txt", timeout=5)

    assert "exit_code=1" in output
    assert "No such file" in output
