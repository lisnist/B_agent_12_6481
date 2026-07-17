from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import BinaryIO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "python_sandbox_runs"
GENERATED_DIR_NAME = "generated_files"
SANDBOX_DIR_NAME = "python_sandbox"
ENTRYPOINT_NAME = "main.py"

MAX_CODE_CHARS = 100_000
MAX_STDIN_CHARS = 100_000
MAX_ARG_COUNT = 20
MAX_ARG_CHARS = 500
DEFAULT_TIMEOUT_SECONDS = 5.0
MAX_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_OUTPUT_CHARS = 6_000
MAX_OUTPUT_CHARS = 20_000
STREAM_READ_CHUNK = 8192


@dataclass
class _StreamCapture:
    max_bytes: int
    chunks: list[bytes] = field(default_factory=list)
    total_bytes: int = 0

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        kept_bytes = sum(len(item) for item in self.chunks)
        remaining = max(0, self.max_bytes - kept_bytes)
        if remaining:
            self.chunks.append(chunk[:remaining])

    def to_text(self, max_chars: int) -> tuple[str, bool]:
        raw = b"".join(self.chunks)
        text = raw.decode("utf-8", errors="replace")
        truncated = self.total_bytes > len(raw) or len(text) > max_chars
        return text[:max_chars], truncated


def _require_text(name: str, value: object, max_chars: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if len(value) > max_chars:
        raise ValueError(f"{name} is too long; max {max_chars} characters")
    return value


def _normalize_timeout(value: int | float | None) -> float:
    if value is None:
        return DEFAULT_TIMEOUT_SECONDS
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be a number")
    timeout = float(value)
    if timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be greater than 0 and at most {MAX_TIMEOUT_SECONDS}")
    return timeout


def _normalize_max_output_chars(value: int | None) -> int:
    if value is None:
        return DEFAULT_MAX_OUTPUT_CHARS
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("max_output_chars must be an integer")
    if value <= 0 or value > MAX_OUTPUT_CHARS:
        raise ValueError(f"max_output_chars must be greater than 0 and at most {MAX_OUTPUT_CHARS}")
    return value


def _normalize_argv(value: list | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("argv must be an array of strings")
    if len(value) > MAX_ARG_COUNT:
        raise ValueError(f"argv must contain at most {MAX_ARG_COUNT} items")
    argv: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise ValueError(f"argv[{index}] must be a string")
        if len(item) > MAX_ARG_CHARS:
            raise ValueError(f"argv[{index}] is too long; max {MAX_ARG_CHARS} characters")
        argv.append(item)
    return argv


def _normalize_export_report(value: bool | None) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ValueError("export_report must be a boolean")
    return value


def _safe_generated_root(output_dir: str | None) -> tuple[Path, Path]:
    output_base = Path(output_dir).resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    generated_root = (output_base / GENERATED_DIR_NAME / SANDBOX_DIR_NAME).resolve()
    try:
        generated_root.relative_to(output_base)
    except ValueError as exc:
        raise ValueError("sandbox output directory escapes output_dir") from exc
    generated_root.mkdir(parents=True, exist_ok=True)
    return output_base, generated_root


def _new_run_dir(generated_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{uuid.uuid4().hex[:12]}"
    run_dir = (generated_root / run_id).resolve()
    try:
        run_dir.relative_to(generated_root)
    except ValueError as exc:
        raise ValueError("sandbox run directory escapes generated root") from exc
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def _minimal_env(run_dir: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in ("SystemRoot", "WINDIR", "COMSPEC"):
        value = os.environ.get(key)
        if value:
            env[key] = value
    temp_dir = str(run_dir)
    env["TEMP"] = temp_dir
    env["TMP"] = temp_dir
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _drain_stream(stream: BinaryIO, capture: _StreamCapture) -> None:
    try:
        while True:
            chunk = stream.read(STREAM_READ_CHUNK)
            if not chunk:
                break
            capture.append(chunk)
    except OSError:
        return


def _write_stdin(stream: BinaryIO, data: bytes) -> None:
    try:
        if data:
            stream.write(data)
            stream.flush()
    except OSError:
        pass
    finally:
        try:
            stream.close()
        except OSError:
            pass


def _process_options(run_dir: Path) -> dict:
    options = {
        "cwd": str(run_dir),
        "env": _minimal_env(run_dir),
        "stdin": subprocess.PIPE,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "shell": False,
    }
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        flags |= getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if flags:
            options["creationflags"] = flags
    else:
        options["start_new_session"] = True
    return options


def _terminate_process_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                shell=False,
                timeout=3,
            )
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except OSError:
            pass
    process.kill()


def _relative_to_output(path: Path, output_base: Path) -> str:
    return path.resolve().relative_to(output_base.resolve()).as_posix()


def _list_files(run_dir: Path, output_base: Path, *, limit: int = 50) -> tuple[list[dict], bool]:
    files: list[dict] = []
    truncated = False
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if len(files) >= limit:
            truncated = True
            break
        files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "relative_output_path": _relative_to_output(path, output_base),
                "num_bytes": path.stat().st_size,
            }
        )
    return files, truncated


def _write_report(report_path: Path, data: dict) -> int:
    content = json.dumps(data, ensure_ascii=False, indent=2)
    report_path.write_text(content, encoding="utf-8", newline="\n")
    return len(content)


def _unique_report_path(run_dir: Path) -> Path:
    candidate = run_dir / "execution_report.json"
    if not candidate.exists():
        return candidate
    index = 1
    while True:
        candidate = run_dir / f"execution_report_{index}.json"
        if not candidate.exists():
            return candidate
        index += 1


def _termination_reason(exit_code: int, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if exit_code == 0:
        return "completed"
    return "nonzero_exit"


def _diagnostic(exit_code: int, timed_out: bool, timeout: float) -> str:
    if timed_out:
        return f"Process exceeded {timeout:g} seconds and was terminated by python_sandbox."
    if exit_code != 0:
        return f"Process exited with non-zero code {exit_code}."
    return "Process completed successfully."


def _result_text(
    exit_code: int,
    timed_out: bool,
    timeout: float,
    stdout: str,
    stderr: str,
) -> str:
    reason = _termination_reason(exit_code, timed_out)
    diagnostic = _diagnostic(exit_code, timed_out, timeout)
    pieces = [f"{reason}: {diagnostic}"]
    if stdout:
        pieces.append(f"stdout: {stdout.strip()[:300]}")
    if stderr:
        pieces.append(f"stderr: {stderr.strip()[:300]}")
    return " | ".join(pieces)


def _hide_entrypoint_path(stderr: str, entrypoint: Path) -> str:
    if not stderr:
        return stderr
    text = stderr.replace(str(entrypoint), ENTRYPOINT_NAME)
    return text.replace(entrypoint.as_posix(), ENTRYPOINT_NAME)


def python_sandbox(
    code: str,
    stdin: str | None = None,
    argv: list | None = None,
    timeout_seconds: int | float | None = None,
    max_output_chars: int | None = None,
    export_report: bool | None = None,
    output_dir: str | None = None,
) -> dict:
    source = _require_text("code", code, MAX_CODE_CHARS)
    stdin_text = _require_text("stdin", "" if stdin is None else stdin, MAX_STDIN_CHARS, allow_empty=True)
    normalized_argv = _normalize_argv(argv)
    timeout = _normalize_timeout(timeout_seconds)
    output_limit = _normalize_max_output_chars(max_output_chars)
    should_export_report = _normalize_export_report(export_report)

    output_base, generated_root = _safe_generated_root(output_dir)
    run_dir = _new_run_dir(generated_root)
    entrypoint = run_dir / ENTRYPOINT_NAME
    entrypoint.write_text(source, encoding="utf-8", newline="\n")

    command = [sys.executable, "-I", "-S", str(entrypoint), *normalized_argv]
    display_command = ["python", "-I", "-S", ENTRYPOINT_NAME, *normalized_argv]
    max_stream_bytes = max(4096, output_limit * 4)
    stdout_capture = _StreamCapture(max_stream_bytes)
    stderr_capture = _StreamCapture(max_stream_bytes)
    stdin_bytes = stdin_text.encode("utf-8")

    start = perf_counter()
    timed_out = False
    process = subprocess.Popen(command, **_process_options(run_dir))

    threads: list[threading.Thread] = []
    if process.stdout is not None:
        threads.append(threading.Thread(target=_drain_stream, args=(process.stdout, stdout_capture), daemon=True))
    if process.stderr is not None:
        threads.append(threading.Thread(target=_drain_stream, args=(process.stderr, stderr_capture), daemon=True))
    if process.stdin is not None:
        threads.append(threading.Thread(target=_write_stdin, args=(process.stdin, stdin_bytes), daemon=True))
    for thread in threads:
        thread.start()

    try:
        exit_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _terminate_process_tree(process)
        try:
            exit_code = process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            exit_code = process.returncode if isinstance(process.returncode, int) else -1

    for thread in threads:
        thread.join(timeout=1)

    duration_ms = round((perf_counter() - start) * 1000, 3)
    stdout_text, stdout_truncated = stdout_capture.to_text(output_limit)
    stderr_text, stderr_truncated = stderr_capture.to_text(output_limit)
    stderr_text = _hide_entrypoint_path(stderr_text, entrypoint)
    termination_reason = _termination_reason(exit_code, timed_out)
    diagnostic = _diagnostic(exit_code, timed_out, timeout)
    result_text = _result_text(exit_code, timed_out, timeout, stdout_text, stderr_text)

    report_path = _unique_report_path(run_dir)
    created_files, created_files_truncated = _list_files(run_dir, output_base)
    report = {
        "language": "python",
        "entrypoint": ENTRYPOINT_NAME,
        "command": display_command,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "timeout_seconds": timeout,
        "duration_ms": duration_ms,
        "termination_reason": termination_reason,
        "diagnostic": diagnostic,
        "text": result_text,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "stdout_truncated": stdout_truncated,
        "stderr_truncated": stderr_truncated,
        "created_files": created_files,
        "created_files_truncated": created_files_truncated,
    }
    num_chars = _write_report(report_path, report)
    num_bytes = report_path.stat().st_size
    relative_report_path = _relative_to_output(report_path, output_base)

    result = {
        **report,
        "python_executable": sys.executable,
        "sandbox_dir": str(run_dir),
        "relative_sandbox_dir": _relative_to_output(run_dir, output_base),
        "code_file_path": str(entrypoint),
        "relative_code_path": _relative_to_output(entrypoint, output_base),
        "report_saved": True,
        "report_exported": should_export_report,
        "report_filename": report_path.name,
        "num_chars": num_chars,
        "num_bytes": num_bytes,
    }
    if should_export_report:
        result.update(
            {
                "generated_file_path": str(report_path),
                "relative_output_path": relative_report_path,
                "filename": report_path.name,
                "file_type": "json",
                "suffix": ".json",
                "overwritten": False,
            }
        )
    return result
