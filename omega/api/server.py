"""HTTP API layer for attachment scan over Omega runtime."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
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
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from omega.api.middleware import RequestDeadlineMiddleware, StreamingBodyLimitMiddleware
from fastapi.responses import JSONResponse

from omega.api.chunk_pipeline import score_chunks
from omega.api.auth import (
    _approval_internal_auth_cfg as _approval_internal_auth_cfg_impl,
    _canonical_request_string as _canonical_request_string_impl,
    _enforce_transport_security as _enforce_transport_security_impl,
    _valid_api_key as _valid_api_key_impl,
    _verify_hmac_request as _verify_hmac_request_impl,
)
from omega.api.incident_export import (
    IncidentApiKeyRecord,
    IncidentApiKeyStore,
    IncidentExportConfig,
    IncidentExportStore,
    IncidentRateLimiter,
    build_incident_record_from_scan,
    key_fingerprint,
)
from omega.api.incident_replay import IncidentReplayConfig, IncidentReplayJobManager, IncidentReplayStore
from omega.api.session_store import ApiSessionStore
from omega.api.request_parsing import (
    _parse_request_payload as _parse_request_payload_impl,
    _parse_session_reset_payload as _parse_session_reset_payload_impl,
)
from omega.api.response_builder import (
    _attestation_block as _attestation_block_impl,
    _audit_log_api_response as _audit_log_api_response_impl,
    _problem_payload as _problem_payload_impl,
    build_api_error_log_event,
    normalize_http_exception_payload,
)
from omega.api.routes.incidents import build_incidents_router
from omega.api.routes.notifications import build_notifications_router
from omega.api.routes.scan import build_scan_router
from omega.api.routes.session import build_session_router
from omega.api.scan_request_orchestration import ScanRequestDeps, run_scan_request
from omega.api.scan_runtime import (
    build_api_risk_event as _build_api_risk_event_impl,
    build_document_scan_report as _build_document_scan_report_impl,
    infer_format as _infer_format_impl,
    monitor_attribution_rows as _monitor_attribution_rows_impl,
    normalize_trust_band as _normalize_trust_band_impl,
    resolve_control_outcome as _resolve_control_outcome_impl,
    source_risk_band as _source_risk_band_impl,
    source_type_for_format as _source_type_for_format_impl,
)
from omega.api.runtime_factory import (
    ApiAttestation,
    ApiAuth,
    ApiDebug,
    ApiLimits,
    ApiLogging,
    ApiRuntime,
    ApiSecurity,
    NonceReplayCache,
    SessionLockPool,
    ScanRuntime,
    _effective_runtime_mode as _effective_runtime_mode_impl,
    _guard_mode as _guard_mode_impl,
    _make_runtime as _make_runtime_impl,
    _parse_api_keys as _parse_api_keys_impl,
    _runtime_config as _runtime_config_impl,
)
from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OmegaState
from omega.log_contract import make_log_event, normalize_api_risk_score
from omega.monitoring.collector import build_monitor_collector_from_config
from omega.monitoring.enrichment import build_downstream_summary, build_redacted_fragments
from omega.monitoring.hints import infer_false_positive_hint
from omega.monitoring.mode import GuardMode, resolve_guard_mode
from omega.monitoring.models import MonitorEvent
from omega.notifications.dispatcher import NotificationDispatcher, build_dispatcher_from_config
from omega.notifications.models import RiskEvent, new_event_id, utc_now_iso
from omega.notifications.security import (
    verify_internal_hmac,
    verify_slack_signature,
    verify_telegram_secret_token,
)
from omega.notifications.startup_flow import run_startup_notifications
from omega.release import get_release_info
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.rag.attachment_ingestion import AttachmentExtractResult, extract_attachment, extract_text_payload
from omega.rag.source_policy import SourceTrustPolicy
from omega.runtime.scan_pipeline import (
    apply_semantic_failure_policy_to_actions,
    compose_control_outcome_state,
    compose_effective_actions,
    evaluate_projection_phase,
    normalize_action_types,
    projection_semantic_failed as shared_projection_semantic_failed,
    run_core_step_phase,
    semantic_failure_policy_from_config as shared_semantic_failure_policy_from_config,
)
from omega.telemetry.ids import build_decision_id, build_trace_id_api
from omega.telemetry.incident_artifact import build_incident_artifact, should_capture_incident_text, should_emit_incident_artifact
from omega.telemetry.anonymous import AnonymousTelemetryService, build_telemetry_event
from omega.structured_logging import build_structured_emitter_from_config, engine_version

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
    return _infer_format_impl(filename=filename, mime=mime)


def _source_type_for_format(fmt: str) -> str:
    return _source_type_for_format_impl(fmt)


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


def _monitor_attribution_rows(*, items: Sequence[ContentItem], top_chunks: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return _monitor_attribution_rows_impl(items=items, top_chunks=top_chunks)


def _normalize_trust_band(value: str) -> str:
    return _normalize_trust_band_impl(value)


def _source_risk_band(items: Sequence[ContentItem]) -> str:
    return _source_risk_band_impl(items)


def _resolve_control_outcome(*, action_types: Sequence[str], verdict: str) -> str:
    return _resolve_control_outcome_impl(action_types=action_types, verdict=verdict)


def _semantic_failure_policy_from_cfg(cfg: Mapping[str, Any]) -> str:
    return str(shared_semantic_failure_policy_from_config(cfg))


def _projection_semantic_failed(projections: Sequence[Any]) -> bool:
    return bool(shared_projection_semantic_failed(projections))


def _valid_api_key(provided: str, configured_keys: Sequence[str]) -> bool:
    return bool(_valid_api_key_impl(provided, configured_keys))


def _parse_api_keys(cfg: Mapping[str, Any]) -> List[str]:
    return list(_parse_api_keys_impl(cfg))


def _make_runtime(resolved_config: Dict[str, Any]) -> ScanRuntime:
    return _make_runtime_impl(resolved_config)


def _runtime_config(runtime: ScanRuntime) -> ApiRuntime:
    return _runtime_config_impl(runtime)


def _guard_mode(runtime: ScanRuntime) -> GuardMode:
    return _guard_mode_impl(runtime)


def _effective_runtime_mode(runtime: ScanRuntime, parsed: Mapping[str, Any]) -> str:
    return _effective_runtime_mode_impl(runtime, parsed)


async def _parse_request_payload(request: Request, limits: ApiLimits) -> Dict[str, Any]:
    return await _parse_request_payload_impl(request, limits)


async def _parse_session_reset_payload(request: Request) -> Dict[str, Any]:
    return await _parse_session_reset_payload_impl(request)


def _request_is_https_proxy_mode(request: Request) -> bool:
    runtime = getattr(getattr(request, "app", None), "state", None)
    scan_runtime = getattr(runtime, "scan_runtime", None)
    if scan_runtime is None:
        return False
    from omega.api.auth import _request_is_https_proxy_mode as _impl
    return bool(_impl(request, scan_runtime.security))


def _enforce_transport_security(request: Request, security: ApiSecurity) -> None:
    _enforce_transport_security_impl(request, security)


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
    return _canonical_request_string_impl(
        method=method,
        path=path,
        body_sha256_hex=body_sha256_hex,
        tenant_id=tenant_id,
        request_id=request_id,
        timestamp=timestamp,
        nonce=nonce,
    )


def _verify_hmac_request(
    *,
    request: Request,
    runtime: ScanRuntime,
    parsed: Dict[str, Any],
    body_bytes: bytes,
    provided_api_key: str,
) -> None:
    _verify_hmac_request_impl(
        request=request,
        runtime=runtime,
        parsed=parsed,
        body_bytes=body_bytes,
        provided_api_key=provided_api_key,
    )


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
    return _attestation_block_impl(
        response_wo_attestation=response_wo_attestation,
        runtime=runtime,
        build_jws_fn=_build_jws_rs256,
    )


def _audit_log_api_response(
    *,
    runtime: ScanRuntime,
    request: Request,
    parsed: Dict[str, Any],
    body_bytes: bytes,
    response_payload: Dict[str, Any],
) -> None:
    _audit_log_api_response_impl(
        runtime=runtime,
        request=request,
        parsed=parsed,
        body_bytes=body_bytes,
        response_payload=response_payload,
    )


def _build_document_scan_report(
    *,
    chunk_agg: Any,
    fmt: str,
    ingestion_flags: Sequence[str],
    max_chunks: int,
) -> Dict[str, Any]:
    return _build_document_scan_report_impl(
        chunk_agg=chunk_agg,
        fmt=fmt,
        ingestion_flags=ingestion_flags,
        max_chunks=max_chunks,
    )


def _notifications_cfg(runtime: ScanRuntime) -> Dict[str, Any]:
    raw = runtime.config.get("notifications", {}) if isinstance(runtime.config, dict) else {}
    return dict(raw or {}) if isinstance(raw, dict) else {}


def _approval_internal_auth_cfg(runtime: ScanRuntime) -> Dict[str, Any]:
    return _approval_internal_auth_cfg_impl(runtime)


def _incident_export_enabled(runtime: ScanRuntime) -> bool:
    cfg = getattr(runtime, "incident_export_cfg", None)
    return bool(cfg is not None and cfg.enabled)


def _incident_replay_enabled(runtime: ScanRuntime) -> bool:
    cfg = getattr(runtime, "incident_replay_cfg", None)
    return bool(cfg is not None and cfg.enabled)


def _incident_contract_version(runtime: ScanRuntime) -> str:
    cfg = getattr(runtime, "incident_export_cfg", None)
    if cfg is None:
        return "1.0"
    return str(cfg.contract_version or "1.0")


def _incident_required_scope(runtime: ScanRuntime) -> str:
    api_cfg = runtime.config.get("api", {}) if isinstance(runtime.config, dict) else {}
    ie_cfg = api_cfg.get("incident_export", {}) if isinstance(api_cfg.get("incident_export", {}), dict) else {}
    auth_cfg = ie_cfg.get("auth", {}) if isinstance(ie_cfg.get("auth", {}), dict) else {}
    return str(auth_cfg.get("required_scope", "incidents:read")).strip() or "incidents:read"


def _incident_replay_scopes(runtime: ScanRuntime) -> Tuple[str, str]:
    cfg = getattr(runtime, "incident_replay_cfg", None)
    if cfg is None:
        return ("incidents:replay:read", "incidents:replay:raw")
    return (str(cfg.required_scope_read), str(cfg.required_scope_raw))


def _problem_payload(*, status_code: int, title: str, detail: str, instance: str) -> Dict[str, Any]:
    return _problem_payload_impl(status_code=status_code, title=title, detail=detail, instance=instance)


def _incident_require_scope(
    *,
    runtime: ScanRuntime,
    provided_key: Optional[str],
    required_scope: str,
) -> Tuple[str, List[str], IncidentApiKeyRecord]:
    if not _incident_export_enabled(runtime):
        raise HTTPException(status_code=404, detail="incident_export_disabled")
    store = runtime.incident_api_key_store
    if store is None:
        raise HTTPException(status_code=503, detail="incident_key_store_unavailable")
    raw = str(provided_key or "").strip()
    if not raw:
        raise HTTPException(status_code=401, detail="unauthorized")
    record = store.resolve_key(provided_key=raw)
    if record is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    scopes = [str(x).strip() for x in list(record.scopes) if str(x).strip()]
    if str(required_scope).strip() not in set(scopes):
        raise HTTPException(status_code=403, detail="forbidden_scope")
    return str(record.key_id), scopes, record


def _incident_require_scopes(
    *,
    runtime: ScanRuntime,
    provided_key: Optional[str],
    required_scopes: Sequence[str],
    feature: str,
) -> Tuple[str, List[str], IncidentApiKeyRecord]:
    if feature == "replay":
        if not _incident_replay_enabled(runtime):
            raise HTTPException(status_code=404, detail="incident_replay_disabled")
    else:
        if not _incident_export_enabled(runtime):
            raise HTTPException(status_code=404, detail="incident_export_disabled")
    store = runtime.incident_api_key_store
    if store is None:
        raise HTTPException(status_code=503, detail="incident_key_store_unavailable")
    raw = str(provided_key or "").strip()
    if not raw:
        raise HTTPException(status_code=401, detail="unauthorized")
    record = store.resolve_key(provided_key=raw)
    if record is None:
        raise HTTPException(status_code=401, detail="unauthorized")
    scopes = [str(x).strip() for x in list(record.scopes) if str(x).strip()]
    scope_set = set(scopes)
    required = [str(x).strip() for x in list(required_scopes) if str(x).strip()]
    for req in required:
        if req not in scope_set:
            raise HTTPException(status_code=403, detail="forbidden_scope")
    return str(record.key_id), scopes, record


def _incident_rate_limit_headers(runtime: ScanRuntime, *, key_id: str) -> Tuple[Dict[str, str], bool]:
    limiter = runtime.incident_rate_limiter
    if limiter is None:
        return ({}, True)
    allowed, remaining, reset_epoch = limiter.check(key_ref=str(key_id))
    headers = {
        "X-RateLimit-Remaining": str(int(remaining)),
        "X-RateLimit-Reset": str(int(reset_epoch)),
    }
    return headers, bool(allowed)


def _replay_access_log(
    *,
    runtime: ScanRuntime,
    key_id: str,
    endpoint: str,
    status_code: int,
    ip: str,
    incident_id: Optional[str],
    job_id: Optional[str],
    scope: str,
    action: str,
) -> None:
    store = runtime.incident_replay_store
    if store is None:
        return
    store.log_access(
        key_hash=key_fingerprint(key_id),
        endpoint=endpoint,
        status_code=int(status_code),
        ip=str(ip),
        incident_id=incident_id,
        job_id=job_id,
        scope=str(scope),
        action=str(action),
    )


def _build_api_risk_event(
    *,
    payload: Mapping[str, Any],
    parsed: Mapping[str, Any],
    fallback_active: bool,
) -> RiskEvent:
    return _build_api_risk_event_impl(payload=payload, parsed=parsed, fallback_active=fallback_active)


def _scan_request(
    runtime: ScanRuntime,
    parsed: Dict[str, Any],
    *,
    include_document_scan_report: bool = False,
) -> Dict[str, Any]:
    return run_scan_request(
        runtime=runtime,
        parsed=parsed,
        include_document_scan_report=include_document_scan_report,
        deps=ScanRequestDeps(
            effective_runtime_mode=_effective_runtime_mode,
            guard_mode=_guard_mode,
            infer_format=_infer_format,
            source_type_for_format=_source_type_for_format,
            resolve_control_outcome=lambda action_types, verdict: _resolve_control_outcome(action_types=action_types, verdict=verdict),
            source_risk_band=_source_risk_band,
            normalize_trust_band=_normalize_trust_band,
            monitor_attribution_rows=lambda items, top_chunks: _monitor_attribution_rows(items=items, top_chunks=top_chunks),
            build_api_risk_event=_build_api_risk_event,
            build_document_scan_report=lambda chunk_agg, fmt, ingestion_flags, max_chunks: _build_document_scan_report(
                chunk_agg=chunk_agg,
                fmt=fmt,
                ingestion_flags=ingestion_flags,
                max_chunks=max_chunks,
            ),
            notifications_cfg=_notifications_cfg,
            incident_export_enabled=_incident_export_enabled,
            incident_replay_enabled=_incident_replay_enabled,
            sha256_hex=_sha256_hex,
            clamp=_clamp,
            attestation_block=_attestation_block,
            score_chunks=score_chunks,
            extract_text_payload=extract_text_payload,
            extract_attachment=extract_attachment,
        ),
    )
def create_app(*, resolved_config: Optional[Dict[str, Any]] = None, profile: str = "dev") -> FastAPI:
    cfg = dict(resolved_config or load_resolved_config(profile=profile).resolved)
    runtime = _make_runtime(cfg)
    runtime_limits = getattr(
        runtime,
        "limits",
        ApiLimits.from_cfg(((cfg.get("api", {}) or {}).get("limits", {}) or {})),
    )
    ocr_cfg = (
        (((cfg.get("retriever", {}) or {}).get("sqlite_fts", {}) or {}).get("attachments", {}) or {}).get("ocr", {})
        or {}
    )

    async def _initialize_ocr_worker(app: FastAPI) -> None:
        enabled = str(ocr_cfg.get("enabled", "false")).strip().lower()
        execution_mode = str(ocr_cfg.get("execution_mode", "inline")).strip().lower()
        prewarm = bool(ocr_cfg.get("prewarm", False))
        if enabled == "false" or execution_mode != "persistent_worker" or not prewarm:
            app.state.ocr_worker_status = {
                "enabled": enabled != "false",
                "status": "lazy" if execution_mode == "persistent_worker" else "inline",
            }
            return
        from omega.vision.ocr_runtime import OCRWorkerSettings, prewarm_ocr_worker

        settings = OCRWorkerSettings(
            provider=str(ocr_cfg.get("provider", "rapidocr")),
            startup_timeout_sec=float(ocr_cfg.get("worker_startup_timeout_sec", 25.0)),
            request_timeout_sec=float(ocr_cfg.get("worker_request_timeout_sec", 15.0)),
            max_memory_mb=int(ocr_cfg.get("worker_max_memory_mb", 2048)),
            max_requests_per_worker=int(ocr_cfg.get("worker_max_requests", 500)),
            pool_size=int(ocr_cfg.get("worker_pool_size", 1)),
            max_pending_requests=int(ocr_cfg.get("worker_max_pending_requests", 2)),
            queue_timeout_sec=float(ocr_cfg.get("worker_queue_timeout_sec", 1.0)),
            intra_op_num_threads=int(ocr_cfg.get("worker_intra_op_threads", 2)),
            inter_op_num_threads=int(ocr_cfg.get("worker_inter_op_threads", 1)),
        )
        try:
            await asyncio.to_thread(prewarm_ocr_worker, settings)
            app.state.ocr_worker_status = {
                "enabled": True,
                "status": "ready",
                "provider": settings.provider,
                "pool_size": settings.pool_size,
                "max_pending_requests": settings.max_pending_requests,
            }
        except Exception as exc:
            app.state.ocr_worker_status = {
                "enabled": True,
                "status": "unavailable",
                "provider": settings.provider,
                "error_type": type(exc).__name__,
            }
            failure_policy = str(ocr_cfg.get("failure_policy", "degrade")).strip().lower()
            if enabled == "true" and failure_policy == "fail_closed":
                raise

    async def _close_runtime_resources(app: FastAPI) -> None:
        current_runtime: ScanRuntime = app.state.scan_runtime
        if current_runtime.telemetry_service is not None:
            current_runtime.telemetry_service.close()
        if current_runtime.notification_dispatcher is not None:
            current_runtime.notification_dispatcher.close()
        if str(ocr_cfg.get("execution_mode", "inline")).strip().lower() == "persistent_worker":
            from omega.vision.ocr_runtime import shutdown_ocr_workers

            await asyncio.to_thread(shutdown_ocr_workers)
        if os.name == "posix":
            from omega.rag.attachment_parser_runtime import shutdown_attachment_parser_broker

            await asyncio.to_thread(shutdown_attachment_parser_broker)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await _initialize_ocr_worker(app)
            yield
        finally:
            await _close_runtime_resources(app)

    app = FastAPI(title="Omega Attachment Scan API", version=get_release_info().engine_version, lifespan=lifespan)
    app.state.scan_runtime = runtime
    app.state.ocr_worker_status = {"enabled": False, "status": "not_configured"}
    app.add_middleware(RequestDeadlineMiddleware, timeout_sec=runtime_limits.request_timeout_sec)
    app.add_middleware(StreamingBodyLimitMiddleware, max_body_bytes=runtime_limits.max_request_body_bytes)
    if _incident_export_enabled(runtime):
        origins = (
            list(runtime.incident_export_cfg.cors_allowed_origins)
            if runtime.incident_export_cfg is not None
            else []
        )
        if origins:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=False,
                allow_methods=["GET", "POST"],
                allow_headers=["X-Omega-API-Key"],
            )
    app.state.startup_summary = run_startup_notifications(
        config=runtime.config,
        profile=str(profile),
        surface="api",
        projector=runtime.projector,
        dispatcher=runtime.notification_dispatcher,
    )

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        runtime: ScanRuntime = app.state.scan_runtime
        emitter = runtime.structured_emitter
        if emitter is not None and emitter.enabled:
            emitter.emit(
                build_api_error_log_event(
                    mode=str(_guard_mode(runtime).value).lower(),
                    path=str(request.url.path),
                    method=str(request.method),
                    status_code=int(exc.status_code),
                    detail=exc.detail,
                )
            )
        payload = normalize_http_exception_payload(
            path=str(request.url.path),
            status_code=int(exc.status_code),
            detail=exc.detail,
        )
        return JSONResponse(status_code=int(exc.status_code), content=payload)

    @app.get("/healthz")
    async def healthz() -> Dict[str, Any]:
        runtime: ScanRuntime = app.state.scan_runtime
        license_status = runtime.license_status
        license_summary: Dict[str, Any] = {"required": bool((runtime.config.get("licensing", {}) or {}).get("required", False))}
        if isinstance(license_status, dict):
            claims = license_status.get("claims", {}) if isinstance(license_status.get("claims", {}), dict) else {}
            license_summary.update({
                "verified": bool(license_status.get("verified", False)),
                "runtime_active": bool(license_status.get("runtime_active", False)),
                "updates_active": bool(license_status.get("updates_active", False)),
                "edition": str(claims.get("edition", "")),
                "features": list(claims.get("features", []) or []),
            })
        return {"status": "ok", "ocr_worker": dict(app.state.ocr_worker_status), "license": license_summary}

    @app.get("/readyz")
    async def readyz() -> Dict[str, Any]:
        runtime: ScanRuntime = app.state.scan_runtime
        failures: list[str] = []
        runtime_cfg = runtime.runtime_cfg
        if runtime_cfg is not None and str(runtime_cfg.mode) == "stateful" and runtime.session_store is None:
            failures.append("session_store_unavailable")
        licensing_required = bool((runtime.config.get("licensing", {}) or {}).get("required", False))
        if licensing_required:
            status = runtime.license_status if isinstance(runtime.license_status, dict) else {}
            if not bool(status.get("verified", False)) or not bool(status.get("runtime_active", False)):
                failures.append("license_inactive")
        required_components = list((runtime.config.get("runtime", {}) or {}).get("required_components", []) or [])
        if "vision" in required_components:
            worker_status = dict(app.state.ocr_worker_status)
            if str(worker_status.get("status", "")).lower() not in {"active", "ready", "ok"}:
                failures.append("ocr_worker_unavailable")
        if failures:
            raise HTTPException(status_code=503, detail={"code": "not_ready", "failures": failures})
        return {"status": "ready", "profile": str(profile), "stateful": runtime.session_store is not None}

    @app.get("/v1/system/version")
    async def system_version() -> Dict[str, Any]:
        return get_release_info().to_dict()

    app.include_router(build_incidents_router())

    @app.get("/v1/monitor/health")
    async def monitor_health() -> Dict[str, Any]:
        runtime: ScanRuntime = app.state.scan_runtime
        collector = runtime.monitor_collector
        snapshot = collector.health_snapshot() if collector is not None else {"enabled": False}
        snapshot["guard_mode"] = str(_guard_mode(runtime).value).lower()
        return snapshot

    app.include_router(
        build_notifications_router(
            notifications_cfg=_notifications_cfg,
            approval_internal_auth_cfg=_approval_internal_auth_cfg,
            valid_api_key=_valid_api_key,
            verify_slack_signature=verify_slack_signature,
            verify_telegram_secret_token=verify_telegram_secret_token,
            verify_internal_hmac=verify_internal_hmac,
        )
    )

    app.include_router(
        build_scan_router(
            enforce_transport_security=_enforce_transport_security,
            valid_api_key=_valid_api_key,
            parse_request_payload=_parse_request_payload,
            verify_hmac_request=_verify_hmac_request,
            effective_runtime_mode=_effective_runtime_mode,
            scan_request=_scan_request,
            audit_log_api_response=_audit_log_api_response,
        )
    )

    app.include_router(build_session_router())

    return app



