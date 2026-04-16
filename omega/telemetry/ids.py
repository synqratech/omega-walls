"""Deterministic telemetry IDs for traceability and replay."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Sequence


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _norm_strings(values: Iterable[Any]) -> list[str]:
    return sorted(str(v).strip() for v in values if str(v).strip())


def build_trace_id_runtime(*, session_id: str, step: int, doc_ids: Sequence[str]) -> str:
    payload = {
        "kind": "runtime",
        "session_id": str(session_id),
        "step": int(step),
        "doc_ids": _norm_strings(doc_ids),
    }
    return f"trc_{_sha256_hex(_stable_json(payload))[:24]}"


def build_trace_id_api(*, tenant_id: str, request_id: str) -> str:
    payload = {
        "kind": "api",
        "tenant_id": str(tenant_id),
        "request_id": str(request_id),
    }
    return f"trc_{_sha256_hex(_stable_json(payload))[:24]}"


def build_decision_id(
    *,
    trace_id: str,
    control_outcome: str,
    action_types: Sequence[str],
    severity: str,
    off: bool,
) -> str:
    payload = {
        "trace_id": str(trace_id),
        "control_outcome": str(control_outcome),
        "action_types": _norm_strings(action_types),
        "severity": str(severity),
        "off": bool(off),
    }
    return f"dec_{_sha256_hex(_stable_json(payload))[:24]}"
