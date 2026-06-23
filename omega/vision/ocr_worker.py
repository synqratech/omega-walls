"""Persistent, local-only OCR worker for untrusted image bytes.

The worker is intentionally a narrow process boundary: it owns the native OCR
runtime and model state, reads bounded requests from stdin, and writes bounded
JSON replies to stdout. It never receives application credentials.
"""
from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
import socket
import sys
from typing import Any

MAX_REQUEST_LINE = 64 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_SPANS = 2048


def _apply_process_hardening(*, max_memory_mb: int) -> None:
    if os.name != "posix":
        return
    import resource

    # ONNX Runtime reserves large virtual address ranges and can segfault under
    # RLIMIT_AS/RLIMIT_DATA even when resident memory is small. The parent applies
    # an RSS watchdog and terminates the worker if it crosses max_memory_mb.
    _ = max_memory_mb
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024 * 1024, 4 * 1024 * 1024))
    if sys.platform.startswith("linux"):
        try:
            libc = ctypes.CDLL(None)
            libc.prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
        except Exception:
            pass


def _disable_network() -> None:
    def _blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise OSError("network disabled in OCR worker")

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


def _polygon(value: Any) -> list[list[float]] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return None
    out: list[list[float]] = []
    for point in value:
        if hasattr(point, "tolist"):
            point = point.tolist()
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        x, y = float(point[0]), float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        out.append([x, y])
    return out if len(out) >= 3 else None


def _confidence(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return max(0.0, min(1.0, result))


def _parse_result(raw_result: Any) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    txts = getattr(raw_result, "txts", None)
    boxes = getattr(raw_result, "boxes", None)
    scores = getattr(raw_result, "scores", None)
    if hasattr(txts, "tolist"):
        txts = txts.tolist()
    if hasattr(boxes, "tolist"):
        boxes = boxes.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(txts, (list, tuple)):
        for idx, item in enumerate(txts[:MAX_SPANS]):
            text = " ".join(str(item).split()).strip()
            if not text:
                continue
            spans.append(
                {
                    "span_id": f"ocr-span-{idx:04d}",
                    "text": text,
                    "confidence": _confidence(scores[idx]) if isinstance(scores, (list, tuple)) and idx < len(scores) else None,
                    "polygon_px": _polygon(boxes[idx]) if isinstance(boxes, (list, tuple)) and idx < len(boxes) else None,
                    "provider_order": idx,
                }
            )
    return spans


def _write(payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > 4 * 1024 * 1024:
        encoded = json.dumps({"ok": False, "error": "response_too_large"}, separators=(",", ":"))
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()


def main() -> int:
    max_memory_mb = int(os.environ.get("OMEGA_OCR_WORKER_MAX_MEMORY_MB", "2048"))
    _apply_process_hardening(max_memory_mb=max_memory_mb)
    _disable_network()
    provider = os.environ.get("OMEGA_OCR_PROVIDER", "rapidocr").strip().lower()
    if provider != "rapidocr":
        _write({"ready": False, "error": "unsupported_provider"})
        return 2
    try:
        from rapidocr import RapidOCR  # type: ignore

        intra_threads = max(1, min(16, int(os.environ.get("OMEGA_OCR_INTRA_OP_THREADS", "2"))))
        inter_threads = max(1, min(8, int(os.environ.get("OMEGA_OCR_INTER_OP_THREADS", "1"))))
        engine = RapidOCR(
            params={
                "EngineConfig.onnxruntime.intra_op_num_threads": intra_threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": inter_threads,
                "Global.log_level": "error",
            }
        )
    except BaseException as exc:
        _write({"ready": False, "error": type(exc).__name__, "detail": str(exc)[:500]})
        return 3
    _write({"ready": True, "provider": provider})

    for raw_line in sys.stdin.buffer:
        if len(raw_line) > MAX_REQUEST_LINE:
            _write({"ok": False, "error": "request_too_large"})
            continue
        try:
            request = json.loads(raw_line.decode("utf-8"))
            request_id = str(request.get("request_id", ""))
            input_path = Path(str(request["input_path"])).resolve(strict=True)
            size = input_path.stat().st_size
            if size <= 0 or size > MAX_IMAGE_BYTES:
                raise ValueError("image_size_out_of_bounds")
            result = engine(str(input_path), use_cls=bool(request.get("use_angle_cls", True)))
            _write({"ok": True, "request_id": request_id, "spans": _parse_result(result)})
        except BaseException as exc:
            _write(
                {
                    "ok": False,
                    "request_id": str(request.get("request_id", "")) if isinstance(locals().get("request"), dict) else "",
                    "error": type(exc).__name__,
                    "detail": str(exc)[:1000],
                }
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
