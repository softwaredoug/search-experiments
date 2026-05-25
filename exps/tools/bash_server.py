from __future__ import annotations

import json
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        started_at = time.perf_counter()
        if self.path != "/execute":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw) if raw else {}
        command = payload.get("command")
        timeout = payload.get("timeout", 30)
        if not isinstance(command, str) or not command.strip():
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Missing command")
            return
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd="/corpus",
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            response = {
                "exit_code": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            response = {
                "exit_code": 124,
                "stdout": "",
                "stderr": "Command timed out",
            }
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        stderr_size = len(response.get("stderr", ""))
        stdout_size = len(response.get("stdout", ""))
        self.log_message(
            "bash_command=%r timeout=%s elapsed_ms=%.2f exit_code=%s stdout_bytes=%s stderr_bytes=%s",
            command,
            timeout,
            elapsed_ms,
            response.get("exit_code"),
            stdout_size,
            stderr_size,
        )
        body = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    server = ThreadingHTTPServer(("0.0.0.0", 8000), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
