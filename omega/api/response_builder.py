"""Response and audit helpers for API server."""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from fastapi import Request
from pydantic import BaseModel

from omega.api.runtime_factory import ScanRuntime
from omega.log_contract import ErrorInfo, make_log_event, normalize_api_risk_score
from omega.structured_logging import engine_version


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _b64url_encode(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _problem_payload(*, status_code: int, title: str, detail: str, instance: str) -> Dict[str, Any]:
    return {
        "type": "about:blank",
        "title": str(title),
        "status": int(status_code),
        "detail": str(detail),
        "instance": str(instance),
    }


class ProblemDetailResponse(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str


def normalize_http_exception_payload(*, path: str, status_code: int, detail: Any) -> Dict[str, Any]:
    if path.startswith("/v1/incidents") or path.startswith("/v1/replay/") or path == "/v1/health":
        if isinstance(detail, dict):
            detail_dict = dict(detail)
            if {"type", "title", "status", "detail", "instance"} <= set(detail_dict.keys()):
                return detail_dict
        return _problem_payload(
            status_code=int(status_code),
            title="Request Failed",
            detail=str(detail),
            instance=str(path),
        )
    return {"detail": detail}


def _build_jws_rs256(*, claims: Mapping[str, Any], kid: str, private_key_pem: str) -> str:
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("cryptography_not_available") from exc

    header = {"alg": "RS256", "typ": "JWT", "kid": kid}
    header_b64 = _b64url_encode(json.dumps(header, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    payload_b64 = _b64url_encode(
        json.dumps(dict(claims), ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    key = serialization.load_pem_private_key(private_key_pem.encode("utf-8"), password=None)
    sig = key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def _attestation_block(
    *,
    response_wo_attestation: Dict[str, Any],
    runtime: ScanRuntime,
    build_jws_fn: Optional[Callable[..., str]] = None,
) -> tuple[Optional[Dict[str, str]], Optional[str]]:
    att_cfg = runtime.attestation
    if not att_cfg.enabled:
        return None, None
    if att_cfg.format != "jws" or att_cfg.alg != "RS256":
        return None, "attestation_unavailable"

    secret_env = att_cfg.private_key_pem_env
    key_pem = str(os.environ.get(secret_env, "")).strip()
    if not key_pem:
        return None, "attestation_unavailable"

    now_i = int(time.time())
    claims = dict(response_wo_attestation)
    claims["iat"] = now_i
    claims["exp"] = now_i + int(att_cfg.exp_sec)
    signer = build_jws_fn or _build_jws_rs256
    try:
        token = signer(claims=claims, kid=att_cfg.kid, private_key_pem=key_pem)
    except Exception:
        return None, "attestation_unavailable"
    return {"alg": "RS256", "kid": att_cfg.kid, "ts": _utc_now(), "jws": token}, None


def _audit_log_api_response(
    *,
    runtime: ScanRuntime,
    request: Request,
    parsed: Dict[str, Any],
    body_bytes: bytes,
    response_payload: Dict[str, Any],
) -> None:
    if not runtime.logging_cfg.enabled:
        return None
    policy_trace = response_payload.get("policy_trace", {}) if isinstance(response_payload.get("policy_trace", {}), dict) else {}
    chunk_trace = policy_trace.get("chunk_pipeline", {}) if isinstance(policy_trace.get("chunk_pipeline", {}), dict) else {}
    filename = str(parsed.get("filename") or "").strip()
    ext = Path(filename).suffix.lower() if filename else ""
    pattern_ids = list(chunk_trace.get("pair_hits", []) or []) + list(chunk_trace.get("rule_ids", []) or [])
    tenant_id = str(parsed.get("tenant_id", ""))
    log_event: Dict[str, Any] = {
        "event": "api_scan_audit",
        "ts": _utc_now(),
        "tenant_id_hash": _sha256_hex(tenant_id),
        "request_id": str(response_payload.get("request_id", parsed.get("request_id", ""))),
        "session_id": str(parsed.get("session_id") or response_payload.get("session_id") or ""),
        "actor_id": str(parsed.get("actor_id") or ""),
        "path": str(request.url.path),
        "method": str(request.method).upper(),
        "mime": str(parsed.get("mime") or ""),
        "filename_ext": ext,
        "payload_size": int(len(body_bytes)),
        "verdict": str(response_payload.get("verdict", "")),
        "control_outcome": str(response_payload.get("control_outcome", "ALLOW")),
        "risk_score": int(response_payload.get("risk_score", 0)),
        "reasons": list(response_payload.get("reasons", []) or []),
        "evidence_id": str(response_payload.get("evidence_id", "")),
        "incident_artifact_id": str(response_payload.get("incident_artifact_id", "")),
        "pattern_ids": sorted(set(pattern_ids)),
    }
    if runtime.logging_cfg.include_policy_trace:
        log_event["policy_trace"] = {
            "off": bool(policy_trace.get("off", False)),
            "severity": str(policy_trace.get("severity", "")),
            "trace_id": str(policy_trace.get("trace_id", "")),
            "decision_id": str(policy_trace.get("decision_id", "")),
            "walls_triggered": list(policy_trace.get("walls_triggered", []) or []),
            "action_types": list(policy_trace.get("action_types", []) or []),
            "chunk_pipeline": {
                "worst_chunk_score": float(chunk_trace.get("worst_chunk_score", 0.0)),
                "pattern_synergy": float(chunk_trace.get("pattern_synergy", 0.0)),
                "confidence": float(chunk_trace.get("confidence", 0.0)),
                "doc_score": float(chunk_trace.get("doc_score", 0.0)),
            },
        }
    import logging

    logging.getLogger("omega.api.server").info("%s", json.dumps(log_event, ensure_ascii=False, sort_keys=True))
    emitter = runtime.structured_emitter
    if emitter is not None and emitter.enabled:
        monitor = dict(response_payload.get("monitor", {}) or {})
        risk_norm, risk_native = normalize_api_risk_score(response_payload.get("risk_score", 0))
        monitor_fragments = list(monitor.get("fragments", []) or [])
        monitor_attr = list(monitor.get("attribution", []) or [])
        if not monitor_attr and isinstance(monitor.get("fragments", []), list):
            monitor_attr = [
                {
                    "source_id": str(x.get("source_id", "")),
                    "doc_id": str(x.get("doc_id", "")),
                    "contribution": float(x.get("contribution", 0.0) or 0.0),
                }
                for x in monitor_fragments
            ]
        emitter.emit(
            make_log_event(
                event="api_scan_audit",
                session_id=(
                    str(response_payload.get("session_id", "")).strip()
                    or str(parsed.get("session_id", "")).strip()
                    or str(response_payload.get("request_id", "")).strip()
                    or "api:unknown"
                ),
                mode=str(monitor.get("guard_mode", "enforce")),
                engine_version=engine_version(),
                risk_score=float(risk_norm),
                intended_action_native=str(monitor.get("intended_action", response_payload.get("control_outcome", "ALLOW"))),
                actual_action_native=str(monitor.get("actual_action", response_payload.get("control_outcome", "ALLOW"))),
                action_types=list((response_payload.get("policy_trace", {}) or {}).get("intended_action_types", []) or []),
                triggered_rules=list(monitor.get("triggered_rules", []) or []),
                attribution_rows=monitor_attr,
                fragments=monitor_fragments,
                fp_hint=(str(monitor.get("false_positive_hint", "")) or None),
                ts=str(log_event.get("ts", _utc_now())),
                trace_id=str(response_payload.get("trace_id", "")),
                decision_id=str(response_payload.get("decision_id", "")),
                surface="api",
                input_type="context_chunk",
                input_length=int(len(body_bytes)),
                source_type=str(parsed.get("mime", "")) or None,
                risk_score_native=risk_native,
            )
        )


def build_api_error_log_event(
    *,
    mode: str,
    path: str,
    method: str,
    status_code: int,
    detail: Any,
) -> Dict[str, Any]:
    return make_log_event(
        event="api_error",
        session_id="api:unknown",
        mode=str(mode),
        engine_version=engine_version(),
        risk_score=0.0,
        intended_action_native="ALLOW",
        actual_action_native="ALLOW",
        action_types=[],
        triggered_rules=[],
        attribution_rows=[],
        ts=_utc_now(),
        surface="api",
        input_type="api_request",
        input_length=None,
        source_type=None,
        error=ErrorInfo(
            code=f"HTTP_{int(status_code)}",
            message=str(detail),
            details={"path": str(path), "method": str(method).upper()},
        ),
    )
