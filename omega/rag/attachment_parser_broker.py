"""Persistent broker that launches one-shot attachment parser sandboxes.

The API process may host native OCR runtimes. Spawning a parser directly after
those runtimes are initialized is not reliable on all libc/Python combinations.
This broker is started before OCR prewarm, owns no application secrets or native
OCR state, and is the only process that forks/execs one-shot document parsers.
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from typing import Any

MAX_REQUEST_LINE = 128 * 1024
MAX_RESPONSE_LINE = 64 * 1024


def _disable_network() -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network disabled in attachment parser broker")

    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    original_socket = socket.socket

    class _NoNetworkSocket(original_socket):
        def connect(self, *_args: Any, **_kwargs: Any) -> Any:
            return _blocked()

        def connect_ex(self, *_args: Any, **_kwargs: Any) -> int:
            _blocked()
            return 1

    socket.socket = _NoNetworkSocket  # type: ignore[assignment]


def _harden() -> None:
    if os.name != "posix":
        return
    import resource

    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None).prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
        except Exception:
            pass


def _write(payload: dict[str, Any]) -> None:
    line = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    if len(line.encode("utf-8")) > MAX_RESPONSE_LINE:
        line = json.dumps(
            {"ok": False, "error": "response_too_large"}, separators=(",", ":")
        )
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _spawn_worker(
    *, worker: Path, request_path: Path, response_path: Path, timeout_sec: float
) -> int:
    """Launch one parser with ``posix_spawn`` and an explicit deadline.

    The broker must remain safe even when the API process has initialized native
    OCR runtimes. Avoiding ``subprocess.Popen`` here also avoids libc at-fork
    hooks and a class of rare pre-exec deadlocks observed under mixed PDF/OCR
    workloads.
    """
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        file_actions = [
            (os.POSIX_SPAWN_DUP2, devnull_fd, 0),
            (os.POSIX_SPAWN_DUP2, devnull_fd, 1),
            (os.POSIX_SPAWN_DUP2, devnull_fd, 2),
        ]
        pid = os.posix_spawn(
            sys.executable,
            [sys.executable, str(worker), str(request_path), str(response_path)],
            dict(os.environ),
            file_actions=file_actions,
            setsid=True,
        )
    finally:
        os.close(devnull_fd)

    deadline = time.monotonic() + max(0.05, float(timeout_sec))
    while True:
        waited_pid, status = os.waitpid(pid, os.WNOHANG)
        if waited_pid == pid:
            return int(os.waitstatus_to_exitcode(status))
        if time.monotonic() >= deadline:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.waitpid(pid, 0)
            except ChildProcessError:
                pass
            raise TimeoutError("parser_timeout")
        time.sleep(0.01)


def main() -> int:
    _harden()
    _disable_network()
    worker = Path(__file__).with_name("attachment_sandbox_worker.py").resolve()
    _write({"ready": True, "pid": os.getpid()})
    for raw_line in sys.stdin.buffer:
        if len(raw_line) > MAX_REQUEST_LINE:
            _write({"ok": False, "error": "request_too_large"})
            continue
        request: dict[str, Any] = {}
        try:
            request = json.loads(raw_line.decode("utf-8"))
            request_id = str(request.get("request_id", ""))
            request_path = Path(str(request["request_path"])).resolve(strict=True)
            response_path = Path(str(request["response_path"])).resolve(strict=False)
            if request_path.parent != response_path.parent:
                raise ValueError("parser paths must share private workdir")
            timeout_sec = max(0.05, min(300.0, float(request["timeout_sec"])))
            try:
                code = _spawn_worker(
                    worker=worker,
                    request_path=request_path,
                    response_path=response_path,
                    timeout_sec=timeout_sec,
                )
            except TimeoutError:
                _write(
                    {"ok": False, "request_id": request_id, "error": "parser_timeout"}
                )
                continue
            _write({"ok": True, "request_id": request_id, "return_code": int(code)})
        except BaseException as exc:
            _write(
                {
                    "ok": False,
                    "request_id": str(request.get("request_id", ""))
                    if isinstance(request, dict)
                    else "",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:1000],
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
