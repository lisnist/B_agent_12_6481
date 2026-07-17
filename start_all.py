from __future__ import annotations

import os
import select
import socket
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
import webbrowser
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
QWEN_API_DIR = PROJECT_ROOT / "llm_backend" / "qwen_api"
MODEL_CONFIG = PROJECT_ROOT / "configs" / "model.yaml"
LOG_DIR = PROJECT_ROOT / "outputs" / "startup_logs"

# Temporary campus-server tunnel settings.
SSH_HOST = "202.199.13.141"
SSH_USER = "root"
SSH_PASSWORD = "KssAT6iTwb"
SSH_PORT = 20115

LOCAL_LLM_HOST = "127.0.0.1"
LOCAL_LLM_PORT = 8012
REMOTE_LLM_HOST = "127.0.0.1"
REMOTE_LLM_PORT = 8012

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8020
FRONTEND_HOST = "127.0.0.1"
FRONTEND_PORT = 5173

LLM_HEALTH_URL = f"http://{LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}/health"
BACKEND_HEALTH_URL = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"
FRONTEND_URL = f"http://{FRONTEND_HOST}:{FRONTEND_PORT}"


class ForwardServer(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class TunnelHandler(socketserver.BaseRequestHandler):
    ssh_transport = None
    remote_host = REMOTE_LLM_HOST
    remote_port = REMOTE_LLM_PORT

    def handle(self) -> None:
        if self.ssh_transport is None or not self.ssh_transport.is_active():
            return
        try:
            channel = self.ssh_transport.open_channel(
                "direct-tcpip",
                (self.remote_host, self.remote_port),
                self.request.getpeername(),
            )
        except Exception:
            return
        if channel is None:
            return
        try:
            while True:
                readable, _, _ = select.select([self.request, channel], [], [])
                if self.request in readable:
                    data = self.request.recv(16384)
                    if not data:
                        break
                    channel.sendall(data)
                if channel in readable:
                    data = channel.recv(16384)
                    if not data:
                        break
                    self.request.sendall(data)
        finally:
            channel.close()
            self.request.close()


@dataclass
class ManagedProcess:
    name: str
    process: subprocess.Popen
    log_file: object


@dataclass
class TunnelRuntime:
    client: object
    server: ForwardServer
    thread: threading.Thread


def is_port_open(host: str, port: int, timeout_seconds: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout_seconds)
        return sock.connect_ex((host, port)) == 0


def http_ready(url: str, timeout_seconds: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_for_url(name: str, url: str, timeout_seconds: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if http_ready(url):
            print(f"{name} ready: {url}", flush=True)
            return
        time.sleep(0.4)
    print(f"{name} started, but health check timed out: {url}", flush=True)


def llm_source() -> str:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("missing dependency: run `pip install PyYAML` first") from exc
    with MODEL_CONFIG.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    runtime = config.get("runtime", {}) if isinstance(config, dict) else {}
    source = runtime.get("llm_source", "local") if isinstance(runtime, dict) else "local"
    if source in {"local", "transformers"}:
        return "local"
    if source in {"fastapi", "api"}:
        return "fastapi"
    if source in {"qwen_api", "qwen", "dashscope"}:
        return "qwen_api"
    raise RuntimeError("configs/model.yaml runtime.llm_source must be local, fastapi, or qwen_api")


def start_tunnel() -> TunnelRuntime | None:
    if http_ready(LLM_HEALTH_URL):
        print(f"LLM tunnel already ready: {LLM_HEALTH_URL}", flush=True)
        return None
    if is_port_open(LOCAL_LLM_HOST, LOCAL_LLM_PORT):
        raise RuntimeError(f"local LLM port is already in use: {LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}")

    try:
        try:
            from cryptography.utils import CryptographyDeprecationWarning

            warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning, module=r"paramiko\..*")
        except Exception:
            pass
        import paramiko
    except ImportError as exc:
        raise RuntimeError("missing dependency: run `pip install paramiko` first") from exc

    print(
        f"opening SSH tunnel: http://{LOCAL_LLM_HOST}:{LOCAL_LLM_PORT} -> "
        f"{SSH_USER}@{SSH_HOST}:{SSH_PORT} -> {REMOTE_LLM_HOST}:{REMOTE_LLM_PORT}",
        flush=True,
    )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=SSH_HOST,
        port=SSH_PORT,
        username=SSH_USER,
        password=SSH_PASSWORD,
        look_for_keys=False,
        allow_agent=False,
        timeout=20,
    )
    transport = client.get_transport()
    if transport is None:
        client.close()
        raise RuntimeError("SSH transport did not initialize")
    transport.set_keepalive(30)

    TunnelHandler.ssh_transport = transport
    server = ForwardServer((LOCAL_LLM_HOST, LOCAL_LLM_PORT), TunnelHandler)
    thread = threading.Thread(target=server.serve_forever, name="llm-ssh-tunnel", daemon=True)
    thread.start()
    wait_for_url("LLM tunnel", LLM_HEALTH_URL, 30)
    return TunnelRuntime(client=client, server=server, thread=thread)


def start_process(name: str, command: list[str], cwd: Path, log_name: str) -> ManagedProcess:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    print(f"starting {name}; log: {log_path}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ManagedProcess(name=name, process=process, log_file=log_file)


def start_backend() -> ManagedProcess | None:
    if http_ready(BACKEND_HEALTH_URL):
        print(f"backend already ready: {BACKEND_HEALTH_URL}", flush=True)
        return None
    process = start_process("backend", [sys.executable, "main.py"], BACKEND_DIR, "backend.log")
    wait_for_url("backend", BACKEND_HEALTH_URL, 30)
    return process


def start_qwen_api_llm() -> ManagedProcess | None:
    if http_ready(LLM_HEALTH_URL):
        print(f"Qwen API LLM already ready: {LLM_HEALTH_URL}", flush=True)
        return None
    if is_port_open(LOCAL_LLM_HOST, LOCAL_LLM_PORT):
        raise RuntimeError(f"local LLM port is already in use: {LOCAL_LLM_HOST}:{LOCAL_LLM_PORT}")
    process = start_process(
        "qwen api llm",
        [sys.executable, "llm_fastapi_server.py"],
        QWEN_API_DIR,
        "qwen_api_llm.log",
    )
    wait_for_url("Qwen API LLM", LLM_HEALTH_URL, 30)
    return process


def start_llm_service() -> tuple[TunnelRuntime | None, ManagedProcess | None]:
    source = llm_source()
    print(f"LLM source: {source}", flush=True)
    if source == "fastapi":
        return start_tunnel(), None
    if source == "qwen_api":
        return None, start_qwen_api_llm()
    print("local LLM source selected; no external LLM service started by start_all.py", flush=True)
    return None, None


def npm_command() -> str:
    return "npm.cmd" if os.name == "nt" else "npm"


def start_frontend() -> ManagedProcess | None:
    if http_ready(FRONTEND_URL):
        print(f"frontend already ready: {FRONTEND_URL}", flush=True)
        return None
    process = start_process(
        "frontend",
        [npm_command(), "run", "dev", "--", "--host", FRONTEND_HOST, "--port", str(FRONTEND_PORT)],
        FRONTEND_DIR,
        "frontend.log",
    )
    wait_for_url("frontend", FRONTEND_URL, 30)
    return process


def stop_process(process: ManagedProcess) -> None:
    if process.process.poll() is None:
        print(f"stopping {process.name}...", flush=True)
        process.process.terminate()
        try:
            process.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.process.kill()
    process.log_file.close()


def stop_tunnel(tunnel: TunnelRuntime | None) -> None:
    if tunnel is None:
        return
    print("stopping SSH tunnel...", flush=True)
    tunnel.server.shutdown()
    tunnel.server.server_close()
    tunnel.client.close()


def main() -> int:
    processes: list[ManagedProcess] = []
    tunnel: TunnelRuntime | None = None
    try:
        tunnel, llm_process = start_llm_service()
        if llm_process is not None:
            processes.append(llm_process)
        backend = start_backend()
        if backend is not None:
            processes.append(backend)
        frontend = start_frontend()
        if frontend is not None:
            processes.append(frontend)
        print(f"opening browser: {FRONTEND_URL}", flush=True)
        webbrowser.open(FRONTEND_URL)
        print("all services started. Press Ctrl+C here to stop processes started by this script.", flush=True)
        while True:
            for process in processes:
                exit_code = process.process.poll()
                if exit_code is not None:
                    raise RuntimeError(f"{process.name} exited with code {exit_code}; see startup log")
            time.sleep(1.0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"fatal: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        for process in reversed(processes):
            stop_process(process)
        stop_tunnel(tunnel)


if __name__ == "__main__":
    raise SystemExit(main())
