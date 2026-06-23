"""Parent-side manager for a persistent, resource-bounded OCR process."""
from __future__ import annotations

import atexit
from dataclasses import dataclass
import json
import os
from pathlib import Path
import queue
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Mapping
import uuid

from omega.vision.contracts import OCRSpan


class OCRWorkerOverloadedError(TimeoutError):
    """Raised when the bounded OCR worker admission queue is saturated."""



@dataclass(frozen=True)
class OCRWorkerSettings:
    provider: str = "rapidocr"
    startup_timeout_sec: float = 25.0
    request_timeout_sec: float = 15.0
    max_memory_mb: int = 2048
    max_requests_per_worker: int = 500
    pool_size: int = 1
    max_pending_requests: int = 2
    queue_timeout_sec: float = 1.0
    intra_op_num_threads: int = 2
    inter_op_num_threads: int = 1

    def __post_init__(self) -> None:
        if self.provider != "rapidocr":
            raise ValueError("persistent OCR worker currently supports rapidocr only")
        if self.startup_timeout_sec <= 0 or self.request_timeout_sec <= 0:
            raise ValueError("OCR worker timeouts must be positive")
        if self.max_memory_mb < 256:
            raise ValueError("OCR worker max_memory_mb must be >= 256")
        if self.max_requests_per_worker < 1:
            raise ValueError("OCR worker max_requests_per_worker must be >= 1")
        if self.pool_size < 1 or self.pool_size > 8:
            raise ValueError("OCR worker pool_size must be in [1,8]")
        if self.max_pending_requests < 0 or self.max_pending_requests > 128:
            raise ValueError("OCR worker max_pending_requests must be in [0,128]")
        if self.queue_timeout_sec <= 0:
            raise ValueError("OCR worker queue_timeout_sec must be positive")
        if self.intra_op_num_threads < 1 or self.intra_op_num_threads > 16:
            raise ValueError("OCR worker intra_op_num_threads must be in [1,16]")
        if self.inter_op_num_threads < 1 or self.inter_op_num_threads > 8:
            raise ValueError("OCR worker inter_op_num_threads must be in [1,8]")


def _worker_env(settings: OCRWorkerSettings) -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "TZ", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMEGA_OCR_PROVIDER"] = settings.provider
    env["OMEGA_OCR_WORKER_MAX_MEMORY_MB"] = str(settings.max_memory_mb)
    env["OMEGA_OCR_INTRA_OP_THREADS"] = str(settings.intra_op_num_threads)
    env["OMEGA_OCR_INTER_OP_THREADS"] = str(settings.inter_op_num_threads)
    # Bound native math runtimes as a second line of defence against CPU
    # starvation of the parent process and timeout watchdog.
    env["OMP_NUM_THREADS"] = str(settings.intra_op_num_threads)
    env["OPENBLAS_NUM_THREADS"] = str(settings.intra_op_num_threads)
    env["MKL_NUM_THREADS"] = str(settings.intra_op_num_threads)
    return env


class OCRWorkerClient:
    def __init__(self, settings: OCRWorkerSettings) -> None:
        self.settings = settings
        self._lock = threading.RLock()
        self._messages: queue.Queue[dict[str, Any] | BaseException] = queue.Queue(maxsize=4)
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._requests = 0

    def _reader_loop(self, proc: subprocess.Popen[str]) -> None:
        """Windows-only pipe reader fallback.

        POSIX uses select()+readline synchronously while the client lock is held,
        so the API process has no persistent OCR reader thread and can safely
        launch independent parser sandboxes after OCR prewarm.
        """
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    self._messages.put(self._decode_line(line), timeout=0.5)
                except queue.Full:
                    return
        except BaseException as exc:
            try:
                self._messages.put_nowait(exc)
            except queue.Full:
                pass

    @staticmethod
    def _decode_line(line: str) -> dict[str, Any]:
        if len(line.encode("utf-8")) > 4 * 1024 * 1024:
            raise RuntimeError("OCR worker response too large")
        try:
            payload = json.loads(line)
        except Exception as exc:
            raise RuntimeError(f"invalid OCR worker response: {exc}") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("invalid OCR worker response type")
        return payload

    def _start_locked(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._stop_locked()
        worker = Path(__file__).with_name("ocr_worker.py")
        self._proc = subprocess.Popen(
            [sys.executable, "-u", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=_worker_env(self.settings),
            close_fds=True,
            start_new_session=True,
        )
        if os.name != "posix":
            self._reader = threading.Thread(
                target=self._reader_loop,
                args=(self._proc,),
                daemon=True,
                name="omega-ocr-reader",
            )
            self._reader.start()
        else:
            self._reader = None
        ready = self._get_message(self.settings.startup_timeout_sec)
        if not isinstance(ready, Mapping) or not bool(ready.get("ready", False)):
            self._stop_locked()
            raise RuntimeError(f"OCR worker failed to start: {ready}")
        self._requests = 0

    def _resident_bytes(self) -> int | None:
        proc = self._proc
        if proc is None or proc.poll() is not None or os.name != "posix":
            return None
        status = Path(f"/proc/{proc.pid}/status")
        try:
            for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _get_message(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout)
        max_resident = int(self.settings.max_memory_mb) * 1024 * 1024
        while True:
            resident = self._resident_bytes()
            if resident is not None and resident > max_resident:
                self._stop_locked()
                raise MemoryError(
                    f"OCR worker resident memory exceeded limit: {resident} > {max_resident}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("OCR worker deadline exceeded")
            proc = self._proc
            if proc is None or proc.poll() is not None:
                raise RuntimeError(
                    f"OCR worker exited unexpectedly: code={None if proc is None else proc.returncode}"
                )
            if os.name == "posix":
                assert proc.stdout is not None
                ready, _, _ = select.select([proc.stdout.fileno()], [], [], min(0.10, remaining))
                if not ready:
                    continue
                line = proc.stdout.readline()
                if not line:
                    raise RuntimeError(f"OCR worker exited unexpectedly: code={proc.poll()}")
                return self._decode_line(line)
            try:
                message = self._messages.get(timeout=min(0.10, remaining))
            except queue.Empty:
                continue
            if isinstance(message, BaseException):
                raise RuntimeError("OCR worker reader failed") from message
            return message

    def _stop_locked(self) -> None:
        proc, self._proc = self._proc, None
        self._reader = None
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
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
        while True:
            try:
                self._messages.get_nowait()
            except queue.Empty:
                break

    def start(self) -> None:
        with self._lock:
            self._start_locked()

    def close(self) -> None:
        with self._lock:
            self._stop_locked()

    def recognize(self, raw: bytes, *, suffix: str, use_angle_cls: bool) -> list[OCRSpan]:
        with self._lock:
            if self._requests >= self.settings.max_requests_per_worker:
                self._stop_locked()
            self._start_locked()
            assert self._proc is not None and self._proc.stdin is not None
            request_id = uuid.uuid4().hex
            with tempfile.TemporaryDirectory(prefix="omega-ocr-input-") as tmp_raw:
                path = Path(tmp_raw).resolve() / f"image{suffix}"
                path.write_bytes(bytes(raw))
                try:
                    path.chmod(0o600)
                except OSError:
                    pass
                payload = {
                    "request_id": request_id,
                    "input_path": str(path),
                    "use_angle_cls": bool(use_angle_cls),
                }
                try:
                    self._proc.stdin.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
                    self._proc.stdin.flush()
                    response = self._get_message(self.settings.request_timeout_sec)
                except BaseException:
                    self._stop_locked()
                    raise
            if str(response.get("request_id", "")) != request_id:
                self._stop_locked()
                raise RuntimeError("OCR worker response correlation mismatch")
            if not bool(response.get("ok", False)):
                raise RuntimeError(f"OCR worker failed: {response.get('error', 'unknown')}")
            spans: list[OCRSpan] = []
            for idx, row in enumerate(list(response.get("spans", []) or [])):
                if not isinstance(row, Mapping):
                    continue
                polygon = row.get("polygon_px")
                spans.append(
                    OCRSpan(
                        span_id=str(row.get("span_id", f"ocr-span-{idx:04d}")),
                        text=str(row.get("text", "")),
                        confidence=(float(row["confidence"]) if row.get("confidence") is not None else None),
                        polygon_px=(tuple((float(p[0]), float(p[1])) for p in polygon) if polygon else None),
                        provider_order=int(row.get("provider_order", idx)),
                    )
                )
            self._requests += 1
            return spans


class OCRWorkerPool:
    """Bounded pool that prevents unbounded request queues around native OCR."""

    def __init__(self, settings: OCRWorkerSettings) -> None:
        self.settings = settings
        self._clients = [OCRWorkerClient(settings) for _ in range(int(settings.pool_size))]
        self._available: queue.Queue[OCRWorkerClient] = queue.Queue(maxsize=len(self._clients))
        for client in self._clients:
            self._available.put_nowait(client)
        self._admission = threading.BoundedSemaphore(
            value=int(settings.pool_size) + int(settings.max_pending_requests)
        )
        self._closed = False
        self._lifecycle_lock = threading.Lock()

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("OCR worker pool is closed")
            started: list[OCRWorkerClient] = []
            try:
                for client in self._clients:
                    client.start()
                    started.append(client)
            except BaseException:
                for client in started:
                    client.close()
                raise

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            for client in self._clients:
                client.close()

    def recognize(self, raw: bytes, *, suffix: str, use_angle_cls: bool) -> list[OCRSpan]:
        if self._closed:
            raise RuntimeError("OCR worker pool is closed")
        if not self._admission.acquire(blocking=False):
            raise OCRWorkerOverloadedError("OCR worker queue is full")
        client: OCRWorkerClient | None = None
        try:
            try:
                client = self._available.get(timeout=float(self.settings.queue_timeout_sec))
            except queue.Empty as exc:
                raise OCRWorkerOverloadedError("OCR worker queue wait deadline exceeded") from exc
            return client.recognize(raw, suffix=suffix, use_angle_cls=use_angle_cls)
        finally:
            if client is not None:
                self._available.put_nowait(client)
            self._admission.release()


_POOLS: dict[OCRWorkerSettings, OCRWorkerPool] = {}
_POOLS_LOCK = threading.Lock()


def _pool_for(settings: OCRWorkerSettings) -> OCRWorkerPool:
    with _POOLS_LOCK:
        pool = _POOLS.get(settings)
        if pool is None:
            pool = OCRWorkerPool(settings)
            _POOLS[settings] = pool
        return pool


def prewarm_ocr_worker(settings: OCRWorkerSettings) -> None:
    # Start the parser broker before native OCR. The broker remains the only
    # process that launches one-shot document parser sandboxes, avoiding fork/exec
    # deadlocks after ONNX Runtime has initialized native state.
    from omega.rag.attachment_parser_runtime import prewarm_attachment_parser_broker

    prewarm_attachment_parser_broker()
    _pool_for(settings).start()


def recognize_with_worker(raw: bytes, *, suffix: str, use_angle_cls: bool, settings: OCRWorkerSettings) -> list[OCRSpan]:
    return _pool_for(settings).recognize(raw, suffix=suffix, use_angle_cls=use_angle_cls)


def shutdown_ocr_workers() -> None:
    with _POOLS_LOCK:
        pools = list(_POOLS.values())
        _POOLS.clear()
    for pool in pools:
        pool.close()


atexit.register(shutdown_ocr_workers)
