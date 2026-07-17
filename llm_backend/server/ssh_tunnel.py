from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request


# Same tunnel as the manual command in 启动说明书.txt:
# ssh -N -L 8012:127.0.0.1:8012 root@<SSH_HOST> -p <SSH_PORT>
SSH_HOST = "202.199.13.141"
SSH_USER = "root"
SSH_PORT = 20115

LOCAL_HOST = "127.0.0.1"
LOCAL_PORT = 8012
REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 8012

HEALTH_PATH = "/health"
HEALTH_TIMEOUT_SECONDS = 20.0
RESTART_DELAY_SECONDS = 3.0
AUTO_RESTART = True


def build_ssh_command() -> list[str]:
    return [
        "ssh",
        "-N",
        "-L",
        f"{LOCAL_PORT}:{REMOTE_HOST}:{REMOTE_PORT}",
        f"{SSH_USER}@{SSH_HOST}",
        "-p",
        str(SSH_PORT),
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=3",
    ]


def local_base_url() -> str:
    return f"http://{LOCAL_HOST}:{LOCAL_PORT}"


def require_ssh_client() -> None:
    if shutil.which("ssh"):
        return
    raise RuntimeError("OpenSSH client was not found in PATH")


def is_port_open(host: str, port: int, timeout_seconds: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_seconds)
        return sock.connect_ex((host, port)) == 0


def check_health(timeout_seconds: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(local_base_url() + HEALTH_PATH, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_until_ready() -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if check_health():
            return True
        time.sleep(0.4)
    return False


def run_tunnel() -> int:
    require_ssh_client()
    if is_port_open(LOCAL_HOST, LOCAL_PORT):
        if check_health():
            print(f"tunnel already ready: {local_base_url()}{HEALTH_PATH}", flush=True)
            return 0
        raise RuntimeError(f"local port {LOCAL_HOST}:{LOCAL_PORT} is already in use")

    command = build_ssh_command()
    while True:
        print(
            f"opening tunnel: {local_base_url()} -> "
            f"{SSH_USER}@{SSH_HOST}:{SSH_PORT} -> {REMOTE_HOST}:{REMOTE_PORT}",
            flush=True,
        )
        process = subprocess.Popen(command)
        if wait_until_ready():
            print(f"tunnel ready: {local_base_url()}{HEALTH_PATH}", flush=True)
        else:
            print("ssh started, but LLM health check did not pass yet", flush=True)

        try:
            exit_code = process.wait()
        except KeyboardInterrupt:
            print("stopping tunnel...", flush=True)
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return 130

        if not AUTO_RESTART:
            return int(exit_code or 0)
        print(f"tunnel exited with code {exit_code}; restarting in {RESTART_DELAY_SECONDS:.1f}s", flush=True)
        time.sleep(RESTART_DELAY_SECONDS)


def main() -> int:
    try:
        return run_tunnel()
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
