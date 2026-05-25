from __future__ import annotations

import atexit
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib import request


class BashService:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.container_id: str | None = None
        self.port: str | None = None
        self.owner_label = f"exps-bash-service-owner={uuid.uuid4().hex}"

    def start(self) -> None:
        server_path = Path(__file__).with_name("bash_server.py")
        command = [
            "docker",
            "run",
            "-d",
            "-p",
            "0:8000",
            "--label",
            "exps-bash-service",
            "--label",
            self.owner_label,
            "-v",
            f"{self.dataset_dir}:/corpus:ro",
            "-v",
            f"{server_path}:/server.py:ro",
            "python:3.12-slim",
            "python",
            "-u",
            "/server.py",
        ]
        try:
            proc = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out starting bash service container.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise RuntimeError(f"Failed to start bash service container. {stderr}") from exc
        self.container_id = proc.stdout.strip()
        try:
            port_proc = subprocess.run(
                ["docker", "port", self.container_id, "8000"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Timed out inspecting bash service port.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else ""
            raise RuntimeError(f"Failed to inspect bash service port. {stderr}") from exc
        port = port_proc.stdout.strip().split(":")[-1]
        if not port:
            raise RuntimeError("Failed to determine docker port mapping")
        self.port = port
        atexit.register(self.stop)

    def stop(self) -> None:
        if not self.container_id:
            return
        subprocess.run(
            ["docker", "rm", "-f", self.container_id],
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.container_id = None
        self.port = None

    def execute(self, command: str, timeout: int = 30) -> str:
        if not self.port:
            raise RuntimeError("Bash service is not running")
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
        return (
            "exit_code="
            + str(exit_code)
            + "\nstdout:\n"
            + stdout
            + "\nstderr:\n"
            + stderr
        )


def start_bash_service(dataset_dir: Path) -> BashService:
    service = BashService(dataset_dir)
    service.start()
    last_exc = None
    for _ in range(10):
        try:
            service.execute("pwd", timeout=5)
            return service
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError("Bash service did not become ready.") from last_exc
