"""Isolated entry point for bounded attachment parsing.

The parent process owns all trust decisions and preflight validation. This worker
runs with a reduced environment, a private working directory and OS resource
limits, and communicates only through bounded JSON files.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
import socket
import ctypes
from pathlib import Path
import importlib.util
import math
import sys
from typing import Any


def _disable_network() -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network disabled in attachment parser sandbox")

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


def _apply_limits(*, max_memory_mb: int, max_cpu_sec: int) -> None:
    if os.name != "posix":
        return
    import resource

    memory = max(64, int(max_memory_mb)) * 1024 * 1024
    # RLIMIT_DATA bounds writable heap without counting mapped shared objects.
    resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))
    cpu_budget = max(1, int(max_cpu_sec))
    usage = resource.getrusage(resource.RUSAGE_SELF)
    consumed_cpu = float(usage.ru_utime) + float(usage.ru_stime)
    # RLIMIT_CPU is an absolute lifetime limit, not a duration from the call.
    # Offset it by trusted worker startup/import CPU so the configured value is
    # a real per-parse budget and does not fail nondeterministically on cold starts.
    cpu_soft = max(1, int(math.ceil(consumed_cpu)) + cpu_budget)
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_soft, cpu_soft + 1))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024 * 1024, 64 * 1024 * 1024))
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None).prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
        except Exception:
            pass


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    ).encode("utf-8")
    path.write_bytes(encoded)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        return 64
    request_path = Path(args[0]).resolve(strict=True)
    response_path = Path(args[1]).resolve(strict=False)
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        # The request and response live in a private TemporaryDirectory owned by
        # the parent. Chdir here keeps relative parser scratch files contained
        # without requiring a fork-based cwd change in the parent process.
        os.chdir(request_path.parent)
        _disable_network()
        # Import trusted parser code before starting the per-request CPU budget.
        # RLIMIT_CPU measures total process lifetime CPU, so counting Python and
        # dependency cold-start time would make an identical attachment pass or
        # fail depending on host cache state.
        parser_path = Path(__file__).with_name("attachment_ingestion.py")
        spec = importlib.util.spec_from_file_location(
            "_omega_attachment_ingestion_worker", parser_path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("unable to load attachment parser module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _apply_limits(
            max_memory_mb=int(request["max_memory_mb"]),
            max_cpu_sec=int(request["max_cpu_sec"]),
        )
        raw = Path(request["input_path"]).resolve(strict=True).read_bytes()
        result = module._extract_attachment_in_process(
            path=None,
            content_bytes=raw,
            filename=request.get("filename"),
            mime=request.get("mime"),
            cfg=request.get("cfg") or {},
        )
        _safe_write_json(response_path, {"ok": True, "result": asdict(result)})
        return 0
    except BaseException as exc:
        try:
            _safe_write_json(
                response_path,
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:2000],
                },
            )
        except BaseException:
            pass
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
