"""Parent-side client for the persistent attachment parser broker."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import select
import signal
import sys
import threading
import time
import uuid
from typing import Any, TextIO


def _broker_env() -> dict[str, str]:
    allowed = {
        "PATH",
        "LANG",
        "LC_ALL",
        "TZ",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
    }
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


class _SpawnedBroker:
    """Minimal ``posix_spawn`` process handle with bounded pipe ownership."""

    def __init__(self, *, pid: int, stdin: TextIO, stdout: TextIO) -> None:
        self.pid = int(pid)
        self.stdin = stdin
        self.stdout = stdout
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is not None:
            return self.returncode
        try:
            waited_pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            self.returncode = -1
            return self.returncode
        if waited_pid == self.pid:
            self.returncode = int(os.waitstatus_to_exitcode(status))
        return self.returncode

    def wait(self, timeout: float) -> int:
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            code = self.poll()
            if code is not None:
                return code
            if time.monotonic() >= deadline:
                raise TimeoutError("attachment parser broker wait deadline exceeded")
            time.sleep(0.01)

    def close_pipes(self) -> None:
        for stream in (self.stdin, self.stdout):
            try:
                stream.close()
            except OSError:
                pass


def _spawn_broker_process() -> _SpawnedBroker:
    """Start the broker without forking the potentially multithreaded API process."""
    broker = Path(__file__).with_name("attachment_parser_broker.py")
    stdin_r, stdin_w = os.pipe()
    stdout_r, stdout_w = os.pipe()
    devnull_fd = os.open(os.devnull, os.O_RDWR)
    try:
        file_actions = [
            (os.POSIX_SPAWN_DUP2, stdin_r, 0),
            (os.POSIX_SPAWN_DUP2, stdout_w, 1),
            (os.POSIX_SPAWN_DUP2, devnull_fd, 2),
            (os.POSIX_SPAWN_CLOSE, stdin_w),
            (os.POSIX_SPAWN_CLOSE, stdout_r),
        ]
        pid = os.posix_spawn(
            sys.executable,
            [sys.executable, "-u", str(broker)],
            _broker_env(),
            file_actions=file_actions,
            setsid=True,
        )
    except BaseException:
        os.close(stdin_w)
        os.close(stdout_r)
        raise
    finally:
        os.close(stdin_r)
        os.close(stdout_w)
        os.close(devnull_fd)
    return _SpawnedBroker(
        pid=pid,
        stdin=os.fdopen(stdin_w, "w", encoding="utf-8", buffering=1),
        stdout=os.fdopen(stdout_r, "r", encoding="utf-8", buffering=1),
    )


class AttachmentParserBrokerClient:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: _SpawnedBroker | None = None
        self._requests = 0
        self._max_requests = 1000

    @staticmethod
    def _decode(line: str) -> dict[str, Any]:
        if len(line.encode("utf-8")) > 64 * 1024:
            raise RuntimeError("attachment parser broker response too large")
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise RuntimeError("invalid attachment parser broker response")
        return payload

    def _read(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + max(0.05, float(timeout))
        while True:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise RuntimeError(
                    f"attachment parser broker exited: {None if proc is None else proc.returncode}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("attachment parser broker deadline exceeded")
            assert proc.stdout is not None
            if os.name == "posix":
                ready, _, _ = select.select(
                    [proc.stdout.fileno()], [], [], min(0.10, remaining)
                )
                if not ready:
                    continue
            line = proc.stdout.readline()
            if line:
                return self._decode(line)

    def _stop_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except OSError:
                pass
            try:
                proc.wait(timeout=1.0)
            except TimeoutError:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except TimeoutError:
                    pass
        proc.close_pipes()

    def _start_locked(self) -> None:
        if (
            self._proc is not None
            and self._proc.poll() is None
            and self._requests < self._max_requests
        ):
            return
        self._stop_locked()
        self._proc = _spawn_broker_process()
        ready = self._read(10.0)
        if not bool(ready.get("ready", False)):
            self._stop_locked()
            raise RuntimeError(f"attachment parser broker failed to start: {ready}")
        self._requests = 0

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def parse(
        self, *, request_path: Path, response_path: Path, timeout_sec: float
    ) -> int:
        with self._lock:
            self._start_locked()
            assert self._proc is not None and self._proc.stdin is not None
            request_id = uuid.uuid4().hex
            payload = {
                "request_id": request_id,
                "request_path": str(request_path),
                "response_path": str(response_path),
                "timeout_sec": float(timeout_sec),
            }
            try:
                self._proc.stdin.write(
                    json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n"
                )
                self._proc.stdin.flush()
                response = self._read(float(timeout_sec) + 2.0)
            except BaseException:
                self._stop_locked()
                raise
            if str(response.get("request_id", "")) != request_id:
                self._stop_locked()
                raise RuntimeError("attachment parser broker correlation mismatch")
            if not bool(response.get("ok", False)):
                error = str(response.get("error", "unknown"))
                if error == "parser_timeout":
                    raise TimeoutError("attachment parser sandbox timeout")
                raise RuntimeError(
                    f"attachment parser broker failed: {error}: {response.get('detail', '')}"
                )
            self._requests += 1
            return int(response.get("return_code", 1))


_BROKER = AttachmentParserBrokerClient()


def prewarm_attachment_parser_broker() -> None:
    if os.name == "posix":
        _BROKER.start()


def run_attachment_parser(
    *, request_path: Path, response_path: Path, timeout_sec: float
) -> int:
    if os.name != "posix":
        raise RuntimeError("attachment parser broker is POSIX-only")
    return _BROKER.parse(
        request_path=request_path, response_path=response_path, timeout_sec=timeout_sec
    )


def shutdown_attachment_parser_broker() -> None:
    _BROKER.close()


atexit.register(shutdown_attachment_parser_broker)
