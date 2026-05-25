from __future__ import annotations

import json
import socket
import subprocess
from pathlib import Path
from urllib import request


class BashService:
    def __init__(self, port: int):
        self.port = port

    def execute(self, command: str, timeout: int = 30) -> str:
        print(f"Executing bash command on service port {self.port}: {command}")
        payload = json.dumps({"command": command, "timeout": timeout}).encode("utf-8")
        req = request.Request(
            f"http://127.0.0.1:{self.port}/execute",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with request.urlopen(req, timeout=timeout + 5) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        exit_code = data.get("exit_code", 1)
        print("Bash command executed with exit code %s. Stdout bytes: %s, Stderr bytes: %s",
              exit_code, len(stdout), len(stderr))
        return (
            "exit_code="
            + str(exit_code)
            + "\nstdout:\n"
            + stdout
            + "\nstderr:\n"
            + stderr
        )


def volume_name_for_dataset(dataset_name: str) -> str:
    return f"exps-bash-corpus-{dataset_name}"


def _volume_exists(volume_name: str) -> bool:
    try:
        subprocess.run(
            ["docker", "volume", "inspect", volume_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return True


def _volume_seeded(volume_name: str) -> bool:
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{volume_name}:/corpus",
                "alpine",
                "sh",
                "-c",
                "test -f /corpus/.exps_seeded",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return True


def ensure_bash_volume(dataset_dir: Path, *, volume_name: str) -> bool:
    created = False
    if not _volume_exists(volume_name):
        subprocess.run(
            ["docker", "volume", "create", volume_name],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        created = True
    if not _volume_seeded(volume_name):
        seed_command = (
            "if [ ! -f /corpus/.exps_seeded ]; then "
            "cp -a /src/. /corpus/ && touch /corpus/.exps_seeded; "
            "fi"
        )
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{volume_name}:/corpus",
                "-v",
                f"{dataset_dir}:/src:ro",
                "alpine",
                "sh",
                "-c",
                seed_command,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
        created = True
    return created


def bash_service_running(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False
