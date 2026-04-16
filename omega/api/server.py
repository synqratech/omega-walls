"""HTTP API layer for attachment scan over Omega runtime."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import json
import logging
import os
from pathlib import Path
import threading
import time
import uuid
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request

from omega.api.chunk_pipeline import score_chunks
from omega.api.session_store import ApiSessionStore
from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OmegaState
from omega.policy.cross_session_state import CrossSessionStateManager
from omega.policy.control_outcome import control_outcome_from_action_types
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.rag.attachment_ingestion import AttachmentExtractResult, extract_attachment, extract_text_payload
from omega.telemetry.ids import build_decision_id, build_trace_id_api
from omega.telemetry.incident_artifact import build_incident_artifact, should_emit_incident_artifact

LOGGER = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_bytes_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64url_decode(data: str) -> bytes:
    payload = str(data or "").strip()
    if not payload:
        return b""
    padding = "=" * ((4 - (len(payload) % 4)) % 4)
    return base64.urlsafe_b64decode((payload + padding).encode("ascii"))


def _infer_format(filename: str | None, mime: str | None) -> str:
    mime_l = str(mime or "").strip().lower()
    if mime_l == "application/pdf":
        return "pdf"
    if mime_l == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return "docx"
    if mime_l == "text/html":
        return "html"
    name = str(filename or "").strip().lower()
    ext = Path(name).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in {".html", ".htm"}:
        return "html"
    if ext == ".zip":
        return "zip"
    return "text"


def _source_type_for_format(fmt: str) -> str:
    if fmt in {"pdf", "docx", "html"}:
        return fmt
    return "other"


def _omega_reason_codes(step_result: Any) -> List[str]:
    out: List[str] = []
    r = step_result.reasons
    if getattr(r, "reason_spike", False):
        out.append("reason_spike")
    if getattr(r, "reason_wall", False):
        out.append("reason_wall")
    if getattr(r, "reason_sum", False):
        out.append("reason_sum")
    if getattr(r, "reason_multi", False):
        out.append("reason_multi")
    return out


def _resolve_control_outcome(*, action_types: Sequence[str], verdict: str) -> str:
    outcome = control_outcome_from_action_types(action_types)
    if outcome != "ALLOW":
        return outcome
    v = str(verdict).strip().lower()
    if v == "block":
        return "SOFT_BLOCK"
    if v == "quarantine":
        return "WARN"
    return "ALLOW"


@dataclass(frozen=True)
class ApiLimits:
    max_file_bytes: int
    max_extracted_text_chars: int
    request_timeout_sec: int

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiLimits":
        data = dict(cfg or {})
        return cls(
            max_file_bytes=int(data.get("max_file_bytes", 20 * 1024 * 1024)),
            max_extracted_text_chars=int(data.get("max_extracted_text_chars", 200_000)),
            request_timeout_sec=int(data.get("request_timeout_sec", 15)),
        )


@dataclass(frozen=True)
class ApiSecurity:
    transport_mode: str
    require_https: bool

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiSecurity":
        data = dict(cfg or {})
        return cls(
            transport_mode=str(data.get("transport_mode", "proxy_tls")).strip().lower(),
            require_https=bool(data.get("require_https", True)),
        )


@dataclass(frozen=True)
class ApiAuth:
    require_hmac: bool
    hmac_secret_env: str
    header_signature: str
    header_timestamp: str
    header_nonce: str
    max_clock_skew_sec: int
    replay_nonce_ttl_sec: int
    replay_cache_max_entries: int

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiAuth":
        data = dict(cfg or {})
        headers = data.get("hmac_headers", {}) if isinstance(data.get("hmac_headers", {}), dict) else {}
        return cls(
            require_hmac=bool(data.get("require_hmac", True)),
            hmac_secret_env=str(data.get("hmac_secret_env", "OMEGA_API_HMAC_SECRET")).strip(),
            header_signature=str(headers.get("signature", "X-Signature")).strip(),
            header_timestamp=str(headers.get("timestamp", "X-Timestamp")).strip(),
            header_nonce=str(headers.get("nonce", "X-Nonce")).strip(),
            max_clock_skew_sec=int(data.get("max_clock_skew_sec", 300)),
            replay_nonce_ttl_sec=int(data.get("replay_nonce_ttl_sec", 600)),
            replay_cache_max_entries=int(data.get("replay_cache_max_entries", 100000)),
        )


@dataclass(frozen=True)
class ApiAttestation:
    enabled: bool
    format: str
    alg: str
    kid: str
    private_key_pem_env: str
    exp_sec: int

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiAttestation":
        data = dict(cfg or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            format=str(data.get("format", "jws")).strip().lower(),
            alg=str(data.get("alg", "RS256")).strip().upper(),
            kid=str(data.get("kid", "omega-attestation-v1")).strip(),
            private_key_pem_env=str(data.get("private_key_pem_env", "OMEGA_API_ATTESTATION_PRIVATE_KEY")).strip(),
            exp_sec=int(data.get("exp_sec", 300)),
        )


@dataclass(frozen=True)
class ApiLogging:
    enabled: bool
    include_policy_trace: bool

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiLogging":
        data = dict(cfg or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            include_policy_trace=bool(data.get("include_policy_trace", True)),
        )


@dataclass(frozen=True)
class ApiDebug:
    enable_document_scan_report: bool
    max_report_chunks: int

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiDebug":
        data = dict(cfg or {})
        return cls(
            enable_document_scan_report=bool(data.get("enable_document_scan_report", False)),
            max_report_chunks=max(1, int(data.get("max_report_chunks", 200))),
        )


@dataclass(frozen=True)
class ApiRuntime:
    mode: str
    allow_request_override: bool
    session_store_backend: str
    session_store_sqlite_path: str
    session_ttl_sec: int
    request_cache_ttl_sec: int

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiRuntime":
        data = dict(cfg or {})
        mode = str(data.get("mode", "stateless")).strip().lower()
        if mode not in {"stateless", "stateful"}:
            raise ValueError("api.runtime.mode must be stateless|stateful")
        session_store = data.get("session_store", {}) if isinstance(data.get("session_store", {}), dict) else {}
        backend = str(session_store.get("backend", "sqlite")).strip().lower()
        if backend != "sqlite":
            raise ValueError("api.runtime.session_store.backend must be sqlite")
        return cls(
            mode=mode,
            allow_request_override=bool(data.get("allow_request_override", True)),
            session_store_backend=backend,
            session_store_sqlite_path=str(session_store.get("sqlite_path", "artifacts/state/api_session_runtime.db")).strip(),
            session_ttl_sec=max(60, int(session_store.get("session_ttl_sec", 86_400))),
            request_cache_ttl_sec=max(60, int(session_store.get("request_cache_ttl_sec", 86_400))),
        )


class SessionLockPool:
    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: Dict[str, asyncio.Lock] = {}

    def get_lock(self, *, tenant_id: str, session_id: str) -> asyncio.Lock:
        key = f"{tenant_id}:{session_id}"
        with self._guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
        return lock


class NonceReplayCache:
    def __init__(self, *, ttl_sec: int, max_entries: int) -> None:
        self.ttl_sec = max(1, int(ttl_sec))
        self.max_entries = max(32, int(max_entries))
        self._entries: Dict[str, float] = {}

    def _cleanup(self, now_ts: float) -> None:
        expired = [k for k, exp in self._entries.items() if exp <= now_ts]
        for key in expired:
            self._entries.pop(key, None)
        if len(self._entries) <= self.max_entries:
            return
        # Bounded in-memory map: drop earliest expirations first.
        sorted_items = sorted(self._entries.items(), key=lambda kv: kv[1])
        overflow = len(self._entries) - self.max_entries
        for key, _ in sorted_items[:overflow]:
            self._entries.pop(key, None)

    def check_and_mark(self, *, key: str, now_ts: float) -> bool:
        self._cleanup(now_ts)
        if key in self._entries:
            return False
        self._entries[key] = now_ts + float(self.ttl_sec)
        return True


@dataclass
class ScanRuntime:
    config: Dict[str, Any]
    projector: Any
    omega_core: OmegaCoreV1
    off_policy: OffPolicyV1
    api_keys: List[str]
    limits: ApiLimits
    security: ApiSecurity
    auth: ApiAuth
    attestation: ApiAttestation
    logging_cfg: ApiLogging
    debug: ApiDebug
    replay_cache: NonceReplayCache
    runtime_cfg: Optional[ApiRuntime] = None
    session_store: Optional[ApiSessionStore] = None
    cross_session: Optional[CrossSessionStateManager] = None
    session_locks: SessionLockPool = field(default_factory=SessionLockPool)


def _valid_api_key(provided: str, configured_keys: Sequence[str]) -> bool:
    for raw in configured_keys:
        item = str(raw).strip()
        if not item:
            continue
        if item.startswith("sha256:"):
            digest = item.split(":", 1)[1].strip().lower()
            if hashlib.sha256(provided.encode("utf-8")).hexdigest() == digest:
                return True
            continue
        if hmac.compare_digest(provided, item):
            return True
    return False


def _parse_api_keys(cfg: Mapping[str, Any]) -> List[str]:
    api_cfg = cfg.get("api", {}) or {}
    auth_cfg = api_cfg.get("auth", {}) or {}
    keys = auth_cfg.get("api_keys", [])
    if not isinstance(keys, list):
        return []
    return [str(x).strip() for x in keys if str(x).strip()]


def _make_runtime(resolved_config: Dict[str, Any]) -> ScanRuntime:
    api_cfg = resolved_config.get("api", {}) or {}
    limits = ApiLimits.from_cfg(api_cfg.get("limits", {}) or {})
    auth_cfg = ApiAuth.from_cfg(api_cfg.get("auth", {}) or {})
    runtime_cfg = ApiRuntime.from_cfg(api_cfg.get("runtime", {}) or {})
    session_store: Optional[ApiSessionStore] = None
    if runtime_cfg.mode == "stateful" or runtime_cfg.allow_request_override:
        session_store = ApiSessionStore(
            sqlite_path=runtime_cfg.session_store_sqlite_path,
            session_ttl_sec=runtime_cfg.session_ttl_sec,
            request_cache_ttl_sec=runtime_cfg.request_cache_ttl_sec,
        )
    return ScanRuntime(
        config=resolved_config,
        projector=build_projector(resolved_config),
        omega_core=OmegaCoreV1(omega_params_from_config(resolved_config)),
        off_policy=OffPolicyV1(resolved_config),
        api_keys=_parse_api_keys(resolved_config),
        limits=limits,
        security=ApiSecurity.from_cfg(api_cfg.get("security", {}) or {}),
        auth=auth_cfg,
        attestation=ApiAttestation.from_cfg(api_cfg.get("attestation", {}) or {}),
        logging_cfg=ApiLogging.from_cfg(api_cfg.get("logging", {}) or {}),
        debug=ApiDebug.from_cfg(api_cfg.get("debug", {}) or {}),
        replay_cache=NonceReplayCache(
            ttl_sec=auth_cfg.replay_nonce_ttl_sec,
            max_entries=auth_cfg.replay_cache_max_entries,
        ),
        runtime_cfg=runtime_cfg,
        session_store=session_store,
        cross_session=CrossSessionStateManager.from_config(resolved_config),
    )


def _runtime_config(runtime: ScanRuntime) -> ApiRuntime:
    if isinstance(runtime.runtime_cfg, ApiRuntime):
        return runtime.runtime_cfg
    api_cfg = runtime.config.get("api", {}) if isinstance(runtime.config.get("api", {}), dict) else {}
    return ApiRuntime.from_cfg(api_cfg.get("runtime", {}) or {})


def _effective_runtime_mode(runtime: ScanRuntime, parsed: Mapping[str, Any]) -> str:
    runtime_cfg = _runtime_config(runtime)
    mode = str(runtime_cfg.mode)
    req_mode = str(parsed.get("runtime_mode", "") or "").strip().lower()
    if runtime_cfg.allow_request_override and req_mode in {"stateless", "stateful"}:
        mode = req_mode
    if mode == "stateful" and not str(parsed.get("session_id") or "").strip():
        raise HTTPException(status_code=400, detail="session_id_required_stateful")
    return mode


async def _parse_request_payload(request: Request, limits: ApiLimits) -> Dict[str, Any]:
    ctype = str(request.headers.get("content-type", "")).lower()
    payload: Dict[str, Any] = {
        "tenant_id": None,
        "request_id": None,
        "session_id": None,
        "actor_id": None,
        "runtime_mode": None,
        "filename": None,
        "mime": None,
        "file_bytes": None,
        "extracted_text": None,
        "input_mode": None,
        "request_id_provided": False,
    }

    if "application/json" in ctype:
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid_json_body") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="invalid_json_body")
        payload["tenant_id"] = body.get("tenant_id")
        payload["request_id"] = body.get("request_id")
        payload["session_id"] = body.get("session_id")
        payload["actor_id"] = body.get("actor_id")
        payload["runtime_mode"] = body.get("runtime_mode")
        payload["filename"] = body.get("filename")
        payload["mime"] = body.get("mime")
        if body.get("extracted_text") is not None:
            payload["extracted_text"] = str(body.get("extracted_text"))
            payload["input_mode"] = "extracted_text"
        if body.get("file_base64") is not None:
            try:
                file_bytes = base64.b64decode(str(body.get("file_base64")), validate=True)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="invalid_file_base64") from exc
            if len(file_bytes) > limits.max_file_bytes:
                raise HTTPException(status_code=413, detail="file_too_large")
            payload["file_bytes"] = file_bytes
            if payload["input_mode"] is None:
                payload["input_mode"] = "file_base64"
    elif "multipart/form-data" in ctype:
        form = await request.form()
        payload["tenant_id"] = form.get("tenant_id")
        payload["request_id"] = form.get("request_id")
        payload["session_id"] = form.get("session_id")
        payload["actor_id"] = form.get("actor_id")
        payload["runtime_mode"] = form.get("runtime_mode")
        payload["filename"] = form.get("filename")
        payload["mime"] = form.get("mime")
        extracted_text = form.get("extracted_text")
        if extracted_text is not None:
            payload["extracted_text"] = str(extracted_text)
            payload["input_mode"] = "extracted_text"
        upload = form.get("file")
        if upload is not None:
            try:
                file_bytes = await upload.read()
            except Exception as exc:
                raise HTTPException(status_code=400, detail="invalid_multipart_file") from exc
            if len(file_bytes) > limits.max_file_bytes:
                raise HTTPException(status_code=413, detail="file_too_large")
            payload["file_bytes"] = file_bytes
            if payload["filename"] in (None, ""):
                payload["filename"] = getattr(upload, "filename", None)
            if payload["mime"] in (None, ""):
                payload["mime"] = getattr(upload, "content_type", None)
            if payload["input_mode"] is None:
                payload["input_mode"] = "file_multipart"
    else:
        raise HTTPException(status_code=415, detail="unsupported_content_type")

    tenant_id = str(payload.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id_required")
    payload["tenant_id"] = tenant_id

    request_id = str(payload.get("request_id") or "").strip() or str(uuid.uuid4())
    payload["request_id_provided"] = bool(str(payload.get("request_id") or "").strip())
    payload["request_id"] = request_id

    session_id = str(payload.get("session_id") or "").strip()
    payload["session_id"] = session_id or None
    actor_id = str(payload.get("actor_id") or "").strip()
    payload["actor_id"] = actor_id or None
    runtime_mode = str(payload.get("runtime_mode") or "").strip().lower()
    if runtime_mode:
        if runtime_mode not in {"stateless", "stateful"}:
            raise HTTPException(status_code=400, detail="invalid_runtime_mode")
        payload["runtime_mode"] = runtime_mode
    else:
        payload["runtime_mode"] = None

    extracted_text = payload.get("extracted_text")
    if extracted_text is not None:
        extracted_text = str(extracted_text)
        if len(extracted_text) > limits.max_extracted_text_chars:
            raise HTTPException(status_code=413, detail="extracted_text_too_large")
        payload["extracted_text"] = extracted_text

    has_extracted = bool(str(payload.get("extracted_text") or "").strip())
    has_file = payload.get("file_bytes") is not None
    if not has_extracted and not has_file:
        raise HTTPException(status_code=400, detail="missing_payload")

    payload["use_extracted_text"] = has_extracted
    return payload


async def _parse_session_reset_payload(request: Request) -> Dict[str, Any]:
    ctype = str(request.headers.get("content-type", "")).lower()
    if "application/json" not in ctype:
        raise HTTPException(status_code=415, detail="unsupported_content_type")
    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid_json_body") from exc
    if not isinstance(body, dict):
        raise HTTPException(status_code=400, detail="invalid_json_body")

    tenant_id = str(body.get("tenant_id") or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=400, detail="tenant_id_required")
    request_id = str(body.get("request_id") or "").strip() or str(uuid.uuid4())
    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id_required")
    actor_id = str(body.get("actor_id") or "").strip() or None
    return {
        "tenant_id": tenant_id,
        "request_id": request_id,
        "session_id": session_id,
        "actor_id": actor_id,
    }


def _request_is_https_proxy_mode(request: Request) -> bool:
    xfp = str(request.headers.get("x-forwarded-proto", "")).strip().lower()
    if xfp:
        first = xfp.split(",")[0].strip()
        if first == "https":
            return True
    forwarded = str(request.headers.get("forwarded", "")).strip()
    if forwarded:
        parts = [p.strip() for p in forwarded.split(";")]
        for part in parts:
            if part.lower().startswith("proto="):
                proto = part.split("=", 1)[1].strip().strip('"').lower()
                if proto == "https":
                    return True
    return False


def _enforce_transport_security(request: Request, security: ApiSecurity) -> None:
    if not security.require_https:
        return
    if security.transport_mode == "proxy_tls":
        if not _request_is_https_proxy_mode(request):
            raise HTTPException(status_code=400, detail="insecure_transport")


def _canonical_request_string(
    *,
    method: str,
    path: str,
    body_sha256_hex: str,
    tenant_id: str,
    request_id: str,
    timestamp: str,
    nonce: str,
) -> str:
    return "\n".join(
        [
            str(method).upper().strip(),
            str(path).strip(),
            str(body_sha256_hex).strip(),
            str(tenant_id).strip(),
            str(request_id).strip(),
            str(timestamp).strip(),
            str(nonce).strip(),
        ]
    )


def _verify_hmac_request(
    *,
    request: Request,
    runtime: ScanRuntime,
    parsed: Dict[str, Any],
    body_bytes: bytes,
    provided_api_key: str,
) -> None:
    auth = runtime.auth
    if not auth.require_hmac:
        return

    signature = str(request.headers.get(auth.header_signature, "")).strip()
    ts_raw = str(request.headers.get(auth.header_timestamp, "")).strip()
    nonce = str(request.headers.get(auth.header_nonce, "")).strip()
    if not signature or not ts_raw or not nonce:
        raise HTTPException(status_code=401, detail="invalid_signature")

    try:
        ts_i = int(ts_raw)
    except Exception as exc:
        raise HTTPException(status_code=401, detail="invalid_signature") from exc
    now_i = int(time.time())
    if abs(now_i - ts_i) > int(auth.max_clock_skew_sec):
        raise HTTPException(status_code=401, detail="stale_timestamp")

    secret_env = auth.hmac_secret_env
    secret = str(os.environ.get(secret_env, "")).strip()
    if not secret:
        raise HTTPException(status_code=401, detail="invalid_signature")

    tenant_id = str(parsed.get("tenant_id") or "").strip()
    request_id = str(parsed.get("request_id") or "").strip()
    body_hash = _sha256_bytes_hex(body_bytes)
    canonical = _canonical_request_string(
        method=request.method,
        path=request.url.path,
        body_sha256_hex=body_hash,
        tenant_id=tenant_id,
        request_id=request_id,
        timestamp=ts_raw,
        nonce=nonce,
    )
    expected_sig = _b64url_encode(hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).digest())
    if not hmac.compare_digest(expected_sig, signature):
        raise HTTPException(status_code=401, detail="invalid_signature")

    replay_key = _sha256_hex(f"{tenant_id}|{hashlib.sha256(provided_api_key.encode('utf-8')).hexdigest()}|{nonce}")
    if not runtime.replay_cache.check_and_mark(key=replay_key, now_ts=float(now_i)):
        raise HTTPException(status_code=409, detail="replay_detected")


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


def _attestation_block(*, response_wo_attestation: Dict[str, Any], runtime: ScanRuntime) -> tuple[Optional[Dict[str, str]], Optional[str]]:
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
    try:
        token = _build_jws_rs256(claims=claims, kid=att_cfg.kid, private_key_pem=key_pem)
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
    filename = str(parsed.get("filename") or "")
    ext = Path(filename).suffix.lower() if filename else ""
    policy_trace = dict(response_payload.get("policy_trace", {}) or {})
    chunk_trace = dict(policy_trace.get("chunk_pipeline", {}) or {})
    top_chunks = chunk_trace.get("top_chunks", [])
    pattern_ids: List[str] = []
    if isinstance(top_chunks, list):
        for row in top_chunks:
            if not isinstance(row, dict):
                continue
            for sig in row.get("pattern_signals", []) or []:
                pattern_ids.append(str(sig))
    log_event = {
        "event": "api_scan_audit",
        "ts": _utc_now(),
        "request_id": str(response_payload.get("request_id", "")),
        "trace_id": str(response_payload.get("trace_id", "")),
        "decision_id": str(response_payload.get("decision_id", "")),
        "tenant_id_hash": _sha256_hex(str(parsed.get("tenant_id", ""))),
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
    LOGGER.info("%s", json.dumps(log_event, ensure_ascii=False, sort_keys=True))


def _build_document_scan_report(
    *,
    chunk_agg: Any,
    fmt: str,
    ingestion_flags: Sequence[str],
    max_chunks: int,
) -> Dict[str, Any]:
    per_chunk: List[Dict[str, Any]] = []
    chunk_scores = list(getattr(chunk_agg, "chunk_scores", []) or [])
    for row in chunk_scores[: max(1, int(max_chunks))]:
        per_chunk.append(
            {
                "chunk_id": str(getattr(row, "doc_id", "")),
                "score_max": float(getattr(row, "score_max", 0.0)),
                "active_walls": list(getattr(row, "active_walls", []) or []),
                "wall_scores": dict(getattr(row, "wall_scores", {}) or {}),
                "pattern_signals": list(getattr(row, "pattern_signals", []) or []),
                "rule_ids": list(getattr(row, "matched_rule_ids", []) or []),
            }
        )
    return {
        "format": str(fmt),
        "chunks_total": int(len(chunk_scores)),
        "chunks_reported": int(len(per_chunk)),
        "ingestion_flags": sorted(set(str(x) for x in ingestion_flags if str(x).strip())),
        "wall_max": dict(getattr(chunk_agg, "wall_max", {}) or {}),
        "worst_chunk_score": float(getattr(chunk_agg, "worst_chunk_score", 0.0)),
        "pattern_synergy": float(getattr(chunk_agg, "pattern_synergy", 0.0)),
        "confidence": float(getattr(chunk_agg, "confidence", 0.0)),
        "doc_score": float(getattr(chunk_agg, "doc_score", 0.0)),
        "pair_hits": list(getattr(chunk_agg, "pair_hits", []) or []),
        "triggered_chunk_ids": list(getattr(chunk_agg, "triggered_chunk_ids", []) or []),
        "rule_ids": list(getattr(chunk_agg, "rule_ids", []) or []),
        "per_chunk": per_chunk,
        "text_included": False,
    }


def _scan_request(
    runtime: ScanRuntime,
    parsed: Dict[str, Any],
    *,
    include_document_scan_report: bool = False,
) -> Dict[str, Any]:
    cfg = runtime.config
    api_cfg = cfg.get("api", {}) or {}
    attachment_cfg = (
        ((cfg.get("retriever", {}) or {}).get("sqlite_fts", {}) or {}).get("attachments", {}) or {}
    )

    filename = str(parsed.get("filename") or "").strip() or None
    mime = str(parsed.get("mime") or "").strip() or None
    tenant_id = str(parsed["tenant_id"])
    request_id = str(parsed["request_id"])
    runtime_mode = _effective_runtime_mode(runtime, parsed)
    session_id = str(parsed.get("session_id") or "").strip() if runtime_mode == "stateful" else None
    actor_id = str(parsed.get("actor_id") or "").strip() if runtime_mode == "stateful" else None
    state_step_prev = 0
    cross_carryover_applied = False
    cross_active_action_types: List[str] = []
    cross_actor_hash: Optional[str] = None
    cross_active_actions: List[Any] = []
    session_store = runtime.session_store
    if runtime_mode == "stateful":
        if session_store is None or not session_id:
            raise HTTPException(status_code=503, detail="stateful_runtime_not_configured")
        cached = session_store.get_cached_response(tenant_id=tenant_id, session_id=session_id, request_id=request_id)
        if cached is not None:
            return cached

    source_id = f"api:{tenant_id}:{request_id}"
    if runtime_mode == "stateful" and session_id:
        source_id = f"api:{tenant_id}:{session_id}"

    ingestion_flags: List[str] = []

    if bool(parsed.get("use_extracted_text", False)):
        extracted = extract_text_payload(text=str(parsed.get("extracted_text") or ""), cfg=attachment_cfg)
        reported_format = _infer_format(filename=filename, mime=mime)
        if reported_format == "zip":
            ingestion_flags.append("zip_deferred_runtime")
        fmt = "text"
    else:
        try:
            extracted = extract_attachment(
                content_bytes=parsed.get("file_bytes"),
                filename=filename,
                mime=mime,
                cfg=attachment_cfg,
            )
        except Exception:
            extracted = AttachmentExtractResult(
                text="",
                chunks=[],
                format=_infer_format(filename=filename, mime=mime),
                text_empty=True,
                scan_like=False,
                hidden_text_chars=0,
                warnings=["ingestion_error", "text_empty"],
                recommended_verdict="quarantine",
            )
        fmt = extracted.format

    ingestion_flags.extend(list(extracted.warnings))
    if fmt == "zip" and "zip_deferred_runtime" not in ingestion_flags:
        ingestion_flags.append("zip_deferred_runtime")

    chunks = [c.text for c in extracted.chunks if str(c.text).strip()]
    if not chunks:
        if extracted.scan_like:
            chunks = ["[attachment_scan_like]"]
        elif extracted.text_empty:
            chunks = ["[attachment_text_empty]"]
        else:
            chunks = ["[attachment_ingestion_empty]"]

    source_type = _source_type_for_format(fmt)
    items: List[ContentItem] = []
    for idx, chunk in enumerate(chunks):
        items.append(
            ContentItem(
                doc_id=f"{request_id}:c{idx:03d}",
                source_id=source_id,
                source_type=source_type,
                trust="untrusted",
                text=str(chunk),
                meta={
                    "tenant_id": tenant_id,
                    "request_id": request_id,
                    "attachment_format": fmt,
                    "ingestion_flags": sorted(set(ingestion_flags)),
                },
            )
        )

    chunk_agg = score_chunks(
        projector=runtime.projector,
        items=items,
        walls=cfg["omega"]["walls"],
        cfg=api_cfg.get("chunk_pipeline", {}) if isinstance(api_cfg.get("chunk_pipeline", {}), dict) else {},
    )
    projections = list(chunk_agg.projections)
    if runtime_mode == "stateful":
        state_vec = np.zeros(4, dtype=float)
        if session_store is None or not session_id:
            raise HTTPException(status_code=503, detail="stateful_runtime_not_configured")
        state_row = session_store.load_session_state(tenant_id=tenant_id, session_id=session_id)
        if state_row is not None:
            state_step_prev = int(state_row.step)
            state_vec = np.asarray(state_row.m, dtype=float)
            if not actor_id:
                actor_id = str(state_row.actor_id)
        if not actor_id:
            actor_id = str(session_id)
        state = OmegaState(
            session_id=f"api:{tenant_id}:{session_id}",
            m=np.asarray(state_vec, dtype=float),
            step=int(state_step_prev),
        )
        if runtime.cross_session is not None:
            hydrated = runtime.cross_session.hydrate_actor_state(actor_id=actor_id, session_id=state.session_id)
            state.m = np.maximum(state.m, np.asarray(hydrated.carried_scars_after_decay, dtype=float))
            cross_carryover_applied = bool(hydrated.carryover_applied)
    else:
        state = OmegaState(session_id=f"api:{tenant_id}:{request_id}", m=np.zeros(4, dtype=float), step=0)
    step_result = runtime.omega_core.step(state=state, items=items, projections=projections)
    decision = runtime.off_policy.select_actions(step_result=step_result, items=items)
    if runtime_mode == "stateful" and session_id and actor_id and runtime.cross_session is not None:
        runtime.cross_session.record_step(
            actor_id=actor_id,
            session_id=state.session_id,
            step_result=step_result,
            policy_actions=decision.actions,
            packet_items=items,
        )
        cross_active_actions = runtime.cross_session.active_actions(
            actor_id=actor_id,
            session_id=state.session_id,
            step=int(step_result.step),
        )
        cross_snapshot = runtime.cross_session.snapshot(
            actor_id=actor_id,
            session_id=state.session_id,
            step=int(step_result.step),
        )
        cross_active_action_types = sorted({str(a.type) for a in cross_active_actions})
        cross_part = (cross_snapshot.get("cross_session", {}) if isinstance(cross_snapshot, dict) else {}) or {}
        cross_carryover_applied = bool(cross_part.get("carryover_applied", cross_carryover_applied))
        actor_hash_val = str(cross_part.get("actor_hash", "")).strip()
        cross_actor_hash = actor_hash_val if actor_hash_val else None

    walls_triggered: List[str] = []
    for idx, wall in enumerate(cfg["omega"]["walls"]):
        if float(step_result.p[idx]) > 0.0 or float(step_result.m_next[idx]) > 0.0:
            walls_triggered.append(str(wall))
    action_types = sorted({str(a.type) for a in list(decision.actions) + list(cross_active_actions)})

    reasons = sorted(
        set(
            _omega_reason_codes(step_result)
            + [str(x) for x in ingestion_flags if str(x).strip()]
            + list(chunk_agg.reasons)
        )
    )
    off = bool(step_result.off)
    severity = str(decision.severity)
    has_exfil = "secret_exfiltration" in walls_triggered
    ingestion_override = any(x in reasons for x in ("scan_like", "text_empty", "zip_deferred_runtime", "ingestion_error"))

    policy_mapper_cfg = api_cfg.get("policy_mapper", {}) if isinstance(api_cfg.get("policy_mapper", {}), dict) else {}
    block_score_threshold = float(policy_mapper_cfg.get("block_score_threshold", 0.72))
    quarantine_score_threshold = float(policy_mapper_cfg.get("quarantine_score_threshold", 0.45))
    quarantine_worst_threshold = float(policy_mapper_cfg.get("quarantine_worst_threshold", 0.38))
    quarantine_synergy_threshold = float(policy_mapper_cfg.get("quarantine_synergy_threshold", 0.20))
    exfil_block_wall_threshold = float(policy_mapper_cfg.get("exfil_block_wall_threshold", 0.60))
    confidence_block_threshold = float(policy_mapper_cfg.get("confidence_block_threshold", 0.55))

    chunk_block = (
        float(chunk_agg.doc_score) >= block_score_threshold
        or (
            float(chunk_agg.wall_max.get("secret_exfiltration", 0.0)) >= exfil_block_wall_threshold
            and float(chunk_agg.confidence) >= confidence_block_threshold
        )
    )
    chunk_quarantine = (
        float(chunk_agg.doc_score) >= quarantine_score_threshold
        or float(chunk_agg.worst_chunk_score) >= quarantine_worst_threshold
        or float(chunk_agg.pattern_synergy) >= quarantine_synergy_threshold
    )

    if ingestion_override:
        verdict = "quarantine"
    elif chunk_block or (off and (severity == "L3" or has_exfil)):
        verdict = "block"
    elif chunk_quarantine or off:
        verdict = "quarantine"
    else:
        verdict = "allow"
    source_quarantine_active = any(
        str(a.type) == "SOURCE_QUARANTINE" and source_id in set(a.source_ids or [])
        for a in cross_active_actions
    )
    tool_freeze_active = any(str(a.type) == "TOOL_FREEZE" for a in cross_active_actions)
    if source_quarantine_active and verdict == "allow":
        verdict = "quarantine"
    if source_quarantine_active:
        reasons.append("source_quarantine_active")
    if tool_freeze_active:
        reasons.append("tool_freeze_active")
    control_outcome = _resolve_control_outcome(action_types=action_types, verdict=verdict)

    max_p = float(np.max(step_result.p)) if len(step_result.p) else 0.0
    sum_m_next = float(np.sum(step_result.m_next))
    off_sigma = float(runtime.omega_core.params.off_Sigma) if float(runtime.omega_core.params.off_Sigma) > 0 else 1.0
    m_ratio = min(1.0, sum_m_next / off_sigma)
    sev = {"L1": 0.0, "L2": 0.5, "L3": 1.0}.get(severity, 0.0)
    risk_legacy = int(round(_clamp(100.0 * (0.60 * max_p + 0.30 * m_ratio + 0.10 * sev), 0.0, 100.0)))
    risk_chunk = int(round(_clamp(100.0 * float(chunk_agg.doc_score), 0.0, 100.0)))
    risk_score = max(risk_legacy, risk_chunk)
    if "zip_deferred_runtime" in reasons:
        risk_score = max(risk_score, 60)
    if ("scan_like" in reasons) or ("text_empty" in reasons):
        risk_score = max(risk_score, 55)

    reasons_sorted = sorted(reasons)
    evidence_id = str(uuid.uuid4())
    trace_id = build_trace_id_api(tenant_id=str(tenant_id), request_id=str(request_id))
    decision_id = build_decision_id(
        trace_id=trace_id,
        control_outcome=control_outcome,
        action_types=action_types,
        severity=severity,
        off=off,
    )
    evidence_summary = {
        "walls_triggered": list(walls_triggered),
        "rule_ids": list(getattr(chunk_agg, "rule_ids", []) or []),
        "chunk_ids": list(getattr(chunk_agg, "triggered_chunk_ids", []) or []),
        "top_chunk_ids": [str(x.get("doc_id", "")) for x in list(chunk_agg.top_chunks)],
        "text_included": False,
        "control_outcome": control_outcome,
        "trace_id": trace_id,
        "decision_id": decision_id,
    }

    payload: Dict[str, Any] = {
        "request_id": request_id,
        "trace_id": trace_id,
        "decision_id": decision_id,
        "tenant_id": tenant_id,
        "risk_score": int(risk_score),
        "verdict": verdict,
        "control_outcome": control_outcome,
        "reasons": reasons_sorted,
        "evidence_id": evidence_id,
        "evidence": evidence_summary,
        "policy_trace": {
            "trace_id": trace_id,
            "decision_id": decision_id,
            "control_outcome": control_outcome,
            "off": off,
            "severity": severity,
            "walls_triggered": walls_triggered,
            "action_types": action_types,
            "max_p": max_p,
            "sum_m_next": sum_m_next,
            "top_docs_count": int(len(step_result.top_docs)),
            "runtime_mode": runtime_mode,
            "state_step_prev": int(state_step_prev),
            "state_step_next": int(step_result.step),
            "ingestion_flags": sorted(set(ingestion_flags)),
            "chunk_pipeline": {
                "chunks_total": int(len(items)),
                "worst_chunk_score": float(chunk_agg.worst_chunk_score),
                "pattern_synergy": float(chunk_agg.pattern_synergy),
                "confidence": float(chunk_agg.confidence),
                "doc_score": float(chunk_agg.doc_score),
                "pair_hits": list(chunk_agg.pair_hits),
                "wall_max": dict(chunk_agg.wall_max),
                "top_chunks": list(chunk_agg.top_chunks),
            },
            "evidence": evidence_summary,
        },
    }
    if runtime_mode == "stateful" and session_id:
        payload["session_id"] = str(session_id)
        payload["policy_trace"]["session_id"] = str(session_id)
        payload["policy_trace"]["cross_session"] = {
            "carryover_applied": bool(cross_carryover_applied),
            "active_action_types": list(cross_active_action_types),
            "actor_hash": cross_actor_hash,
        }
    if should_emit_incident_artifact(config=runtime.config, control_outcome=control_outcome):
        item_by_id = {item.doc_id: item for item in items}
        top_chunk_ids = [str(x) for x in list(evidence_summary.get("top_chunk_ids", []) or []) if str(x).strip()]
        if not top_chunk_ids:
            top_chunk_ids = [str(x) for x in list(evidence_summary.get("chunk_ids", []) or []) if str(x).strip()]
        top_docs = []
        for doc_id in top_chunk_ids:
            item = item_by_id.get(doc_id)
            if item is None:
                continue
            top_docs.append(
                {
                    "doc_id": item.doc_id,
                    "source_id": item.source_id,
                    "source_type": item.source_type,
                    "trust": item.trust,
                    "text": item.text,
                }
            )
        blocked_doc_ids: List[str] = []
        if verdict in {"block", "quarantine"}:
            blocked_doc_ids = [doc_id for doc_id in top_chunk_ids if doc_id in item_by_id]
        quarantined_source_ids = sorted(
            {
                str(source_item)
                for action in cross_active_actions
                if str(getattr(action, "type", "")) == "SOURCE_QUARANTINE"
                for source_item in list(getattr(action, "source_ids", []) or [])
                if str(source_item).strip()
            }
        )
        if verdict == "quarantine":
            quarantined_source_ids = sorted(set(quarantined_source_ids) | {str(source_id)})
        config_sha = _sha256_hex(json.dumps(runtime.config, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
        incident_artifact = build_incident_artifact(
            config=runtime.config,
            surface="api",
            session_id=str(session_id) if session_id else f"req:{request_id}",
            step=int(step_result.step),
            request_id=request_id,
            control_outcome=control_outcome,
            off=off,
            severity=severity,
            verdict=verdict,
            actions=list(decision.actions) + list(cross_active_actions),
            reason_flags=reasons_sorted,
            contributing_signals={
                "max_p": max_p,
                "sum_m_next": sum_m_next,
                "walls_triggered": list(walls_triggered),
                "chunk_pipeline": {
                    "doc_score": float(chunk_agg.doc_score),
                    "worst_chunk_score": float(chunk_agg.worst_chunk_score),
                    "pattern_synergy": float(chunk_agg.pattern_synergy),
                    "confidence": float(chunk_agg.confidence),
                    "wall_max": dict(chunk_agg.wall_max),
                },
            },
            top_docs=top_docs,
            blocked_doc_ids=blocked_doc_ids,
            quarantined_source_ids=quarantined_source_ids,
            context_total_docs=len(items),
            context_allowed_docs=max(0, len(items) - len(set(blocked_doc_ids))),
            evidence_id=evidence_id,
            config_refs={
                "api_config_sha256": config_sha,
                "policy_version": str((runtime.config.get("off_policy", {}) or {}).get("policy_version", "")),
            },
            refs={
                "source_id": str(source_id),
                "cross_active_action_types": list(cross_active_action_types),
                "cross_actor_hash": cross_actor_hash,
            },
            trace_id=trace_id,
            decision_id=decision_id,
        )
        payload["incident_artifact_id"] = str(incident_artifact.get("incident_artifact_id", ""))
        payload["incident_artifact"] = incident_artifact

    att, att_reason = _attestation_block(response_wo_attestation=payload, runtime=runtime)
    if att is not None:
        payload["attestation"] = att
    elif att_reason:
        reasons_sorted = sorted(set(reasons_sorted + [att_reason]))
        payload["reasons"] = reasons_sorted
        payload["policy_trace"]["attestation_status"] = att_reason
    if include_document_scan_report:
        payload["document_scan_report"] = _build_document_scan_report(
            chunk_agg=chunk_agg,
            fmt=fmt,
            ingestion_flags=ingestion_flags,
            max_chunks=runtime.debug.max_report_chunks,
        )
    if runtime_mode == "stateful" and session_id and actor_id and session_store is not None:
        session_store.save_session_state(
            tenant_id=tenant_id,
            session_id=session_id,
            actor_id=actor_id,
            m=np.asarray(step_result.m_next, dtype=float),
            step=int(step_result.step),
        )
        session_store.save_cached_response(
            tenant_id=tenant_id,
            session_id=session_id,
            request_id=request_id,
            response_payload=payload,
        )
    return payload


def create_app(*, resolved_config: Optional[Dict[str, Any]] = None, profile: str = "dev") -> FastAPI:
    cfg = dict(resolved_config or load_resolved_config(profile=profile).resolved)
    app = FastAPI(title="Omega Attachment Scan API", version="1.0")
    app.state.scan_runtime = _make_runtime(cfg)

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/scan/attachment")
    async def scan_attachment(
        request: Request,
        debug: bool = False,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        runtime: ScanRuntime = app.state.scan_runtime
        _enforce_transport_security(request=request, security=runtime.security)
        if not x_api_key or not _valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        body_bytes = await request.body()
        parsed = await _parse_request_payload(request=request, limits=runtime.limits)
        _verify_hmac_request(
            request=request,
            runtime=runtime,
            parsed=parsed,
            body_bytes=body_bytes,
            provided_api_key=str(x_api_key),
        )
        if debug and not runtime.debug.enable_document_scan_report:
            raise HTTPException(status_code=403, detail="debug_mode_disabled")
        mode = _effective_runtime_mode(runtime, parsed)
        if mode == "stateful":
            lock = runtime.session_locks.get_lock(tenant_id=str(parsed["tenant_id"]), session_id=str(parsed["session_id"]))
            async with lock:
                payload = _scan_request(
                    runtime=runtime,
                    parsed=parsed,
                    include_document_scan_report=bool(debug),
                )
        else:
            payload = _scan_request(
                runtime=runtime,
                parsed=parsed,
                include_document_scan_report=bool(debug),
            )
        _audit_log_api_response(
            runtime=runtime,
            request=request,
            parsed=parsed,
            body_bytes=body_bytes,
            response_payload=payload,
        )
        return payload

    @app.post("/v1/scan/attachment/document_scan_report")
    async def scan_attachment_document_report(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        runtime: ScanRuntime = app.state.scan_runtime
        if not runtime.debug.enable_document_scan_report:
            raise HTTPException(status_code=403, detail="debug_mode_disabled")
        _enforce_transport_security(request=request, security=runtime.security)
        if not x_api_key or not _valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        body_bytes = await request.body()
        parsed = await _parse_request_payload(request=request, limits=runtime.limits)
        _verify_hmac_request(
            request=request,
            runtime=runtime,
            parsed=parsed,
            body_bytes=body_bytes,
            provided_api_key=str(x_api_key),
        )
        mode = _effective_runtime_mode(runtime, parsed)
        if mode == "stateful":
            lock = runtime.session_locks.get_lock(tenant_id=str(parsed["tenant_id"]), session_id=str(parsed["session_id"]))
            async with lock:
                payload = _scan_request(runtime=runtime, parsed=parsed, include_document_scan_report=True)
        else:
            payload = _scan_request(runtime=runtime, parsed=parsed, include_document_scan_report=True)
        _audit_log_api_response(
            runtime=runtime,
            request=request,
            parsed=parsed,
            body_bytes=body_bytes,
            response_payload=payload,
        )
        return payload

    @app.post("/v1/session/reset")
    async def reset_session(
        request: Request,
        x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    ) -> Dict[str, Any]:
        runtime: ScanRuntime = app.state.scan_runtime
        _enforce_transport_security(request=request, security=runtime.security)
        if not x_api_key or not _valid_api_key(str(x_api_key), runtime.api_keys):
            raise HTTPException(status_code=401, detail="unauthorized")
        body_bytes = await request.body()
        parsed = await _parse_session_reset_payload(request=request)
        _verify_hmac_request(
            request=request,
            runtime=runtime,
            parsed=parsed,
            body_bytes=body_bytes,
            provided_api_key=str(x_api_key),
        )
        if runtime.session_store is None:
            raise HTTPException(status_code=503, detail="stateful_runtime_not_configured")
        lock = runtime.session_locks.get_lock(tenant_id=str(parsed["tenant_id"]), session_id=str(parsed["session_id"]))
        async with lock:
            existed = runtime.session_store.clear_session(
                tenant_id=str(parsed["tenant_id"]),
                session_id=str(parsed["session_id"]),
            )
        return {
            "request_id": str(parsed["request_id"]),
            "tenant_id": str(parsed["tenant_id"]),
            "session_id": str(parsed["session_id"]),
            "reset": True,
            "existed": bool(existed),
        }

    return app
