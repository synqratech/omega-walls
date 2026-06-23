"""Runtime/bootstrap assembly for Omega API server."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

from fastapi import HTTPException
from omega.edition import verify_runtime_license_if_available
from omega.api.incident_export import (
    IncidentApiKeyStore,
    IncidentExportConfig,
    IncidentExportStore,
    IncidentRateLimiter,
)
from omega.api.incident_replay import (
    IncidentReplayConfig,
    IncidentReplayJobManager,
    IncidentReplayStore,
)
from omega.api.session_store import ApiSessionStore
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.monitoring.collector import MonitorEventCollector, build_monitor_collector_from_config
from omega.monitoring.mode import GuardMode, resolve_guard_mode
from omega.notifications.dispatcher import NotificationDispatcher, build_dispatcher_from_config
from omega.policy.cross_session_state import CrossSessionStateManager
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector
from omega.telemetry.anonymous import AnonymousTelemetryService
from omega.runtime.environment import read_secret_list, read_secret_value
from omega.runtime.skillbox import SkillBox
from omega.structured_logging import StructuredLogEmitter, build_structured_emitter_from_config


@dataclass(frozen=True)
class ApiLimits:
    max_file_bytes: int
    max_extracted_text_chars: int
    request_timeout_sec: int
    max_request_body_bytes: int = 24 * 1024 * 1024
    max_multipart_files: int = 1
    max_multipart_fields: int = 16
    max_multipart_part_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiLimits":
        data = dict(cfg or {})
        return cls(
            max_file_bytes=int(data.get("max_file_bytes", 20 * 1024 * 1024)),
            max_extracted_text_chars=int(data.get("max_extracted_text_chars", 200_000)),
            request_timeout_sec=int(data.get("request_timeout_sec", 15)),
            max_request_body_bytes=int(data.get("max_request_body_bytes", 24 * 1024 * 1024)),
            max_multipart_files=max(1, int(data.get("max_multipart_files", 1))),
            max_multipart_fields=max(1, int(data.get("max_multipart_fields", 16))),
            max_multipart_part_bytes=int(data.get("max_multipart_part_bytes", data.get("max_file_bytes", 20 * 1024 * 1024))),
        )


@dataclass(frozen=True)
class ApiSecurity:
    transport_mode: str
    require_https: bool
    trusted_proxy_cidrs: List[str] = field(default_factory=list)

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ApiSecurity":
        data = dict(cfg or {})
        return cls(
            transport_mode=str(data.get("transport_mode", "proxy_tls")).strip().lower(),
            require_https=bool(data.get("require_https", True)),
            trusted_proxy_cidrs=[str(x).strip() for x in list(data.get("trusted_proxy_cidrs", [])) if str(x).strip()],
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
    # ponytail: process-local replay cache; replace with a shared/durable nonce
    # store before enabling multi-worker or multi-replica production deployment.
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
    notification_dispatcher: Optional[NotificationDispatcher] = None
    monitor_collector: Optional[MonitorEventCollector] = None
    structured_emitter: Optional[StructuredLogEmitter] = None
    incident_export_cfg: Optional[IncidentExportConfig] = None
    incident_export_store: Optional[IncidentExportStore] = None
    incident_api_key_store: Optional[IncidentApiKeyStore] = None
    incident_rate_limiter: Optional[IncidentRateLimiter] = None
    incident_replay_cfg: Optional[IncidentReplayConfig] = None
    incident_replay_store: Optional[IncidentReplayStore] = None
    incident_replay_manager: Optional[IncidentReplayJobManager] = None
    telemetry_service: Optional[AnonymousTelemetryService] = None
    license_status: Optional[Dict[str, Any]] = None
    skillbox: Optional[SkillBox] = None
    session_locks: SessionLockPool = field(default_factory=SessionLockPool)


_WEAK_API_KEYS = {
    "dev-api-key",
    "test-api-key",
    "quickstart-api-key",
    "changeme",
    "change-me",
    "secret",
}


def _is_prod_config(cfg: Mapping[str, Any]) -> bool:
    profile_env = str(((cfg.get("profiles", {}) or {}).get("env", ""))).strip().lower()
    return profile_env in {
        "prod",
        "production",
        "prod_api",
        "prod_vision",
        "prod_vision_local_ocr",
        "prod_enterprise",
        "prod_vision_enterprise",
    }


_REQUIRED_COMPONENT_MODULES: Dict[str, tuple[str, ...]] = {
    "attachments": ("pypdf", "docx", "bs4", "lxml", "PIL"),
    "vision": ("rapidocr", "onnxruntime"),
}


def _validate_required_runtime_components(cfg: Mapping[str, Any]) -> None:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime", {}), Mapping) else {}
    required = runtime_cfg.get("required_components", []) if isinstance(runtime_cfg, Mapping) else []
    if not isinstance(required, list):
        raise RuntimeError("runtime.required_components must be a list")
    unknown = [str(name) for name in required if str(name) not in _REQUIRED_COMPONENT_MODULES]
    if unknown:
        raise RuntimeError(f"unknown required runtime components: {', '.join(unknown)}")
    missing: Dict[str, List[str]] = {}
    for component in required:
        modules = _REQUIRED_COMPONENT_MODULES[str(component)]
        absent = [module for module in modules if importlib.util.find_spec(module) is None]
        if absent:
            missing[str(component)] = absent
    if missing:
        detail = "; ".join(f"{component}={','.join(modules)}" for component, modules in missing.items())
        raise RuntimeError(
            "required production components are unavailable: " + detail +
            '. Install the matching extra, e.g. "omega-walls[api,attachments]" '
            'or "omega-walls[api,vision]".'
        )


def _verify_runtime_license_if_available(resolved_config: Mapping[str, Any]) -> Any:
    """Backward-compatible local wrapper around the centralized edition guard."""
    return verify_runtime_license_if_available(resolved_config)


def _parse_api_keys(cfg: Mapping[str, Any]) -> List[str]:
    api_cfg = cfg.get("api", {}) or {}
    auth_cfg = api_cfg.get("auth", {}) or {}
    keys = auth_cfg.get("api_keys", [])
    if not isinstance(keys, list):
        raise ValueError("api.auth.api_keys must be a list")
    resolved = [str(x).strip() for x in keys if str(x).strip()]
    env_name = str(auth_cfg.get("api_key_env", "OMEGA_API_KEYS")).strip()
    if env_name:
        resolved.extend(read_secret_list(env_name))
    # Preserve order while deduplicating.
    resolved = list(dict.fromkeys(resolved))
    if _is_prod_config(cfg):
        invalid = [key for key in resolved if key.lower() in _WEAK_API_KEYS or (not key.startswith("sha256:") and len(key) < 24)]
        if invalid:
            raise RuntimeError("production API refuses development, placeholder, or short credentials")
        if not resolved:
            raise RuntimeError(f"production API requires credentials via {env_name or 'api.auth.api_keys'}")
    return resolved


def _validate_prod_runtime_secrets(cfg: Mapping[str, Any], auth: ApiAuth, api_keys: List[str]) -> None:
    """Fail before runtime side effects when production credentials are absent or weak."""
    if not _is_prod_config(cfg):
        return
    if not api_keys:
        raise RuntimeError("production API requires at least one API credential")
    if auth.require_hmac:
        env_name = str(auth.hmac_secret_env).strip()
        try:
            secret = read_secret_value(env_name, required=True)
        except ValueError as exc:
            raise RuntimeError(f"production API requires HMAC secret via {env_name} or {env_name}_FILE") from exc
        if secret.lower() in _WEAK_API_KEYS or len(secret) < 32:
            raise RuntimeError("production API refuses development, placeholder, or short HMAC secrets")
        plain_api_keys = {key for key in api_keys if not key.startswith("sha256:")}
        if secret in plain_api_keys:
            raise RuntimeError("production API requires separate API-key and HMAC secrets")


def _make_runtime(resolved_config: Dict[str, Any]) -> ScanRuntime:
    verified_license = _verify_runtime_license_if_available(resolved_config)
    _validate_required_runtime_components(resolved_config)
    api_cfg = resolved_config.get("api", {}) or {}
    limits = ApiLimits.from_cfg(api_cfg.get("limits", {}) or {})
    auth_cfg = ApiAuth.from_cfg(api_cfg.get("auth", {}) or {})
    api_keys = _parse_api_keys(resolved_config)
    _validate_prod_runtime_secrets(resolved_config, auth_cfg, api_keys)
    runtime_cfg = ApiRuntime.from_cfg(api_cfg.get("runtime", {}) or {})
    session_store: Optional[ApiSessionStore] = None
    if runtime_cfg.mode == "stateful" or runtime_cfg.allow_request_override:
        session_store = ApiSessionStore(
            sqlite_path=runtime_cfg.session_store_sqlite_path,
            session_ttl_sec=runtime_cfg.session_ttl_sec,
            request_cache_ttl_sec=runtime_cfg.request_cache_ttl_sec,
        )
    guard_mode = resolve_guard_mode(resolved_config)
    notification_dispatcher = build_dispatcher_from_config(config=resolved_config)
    monitor_collector = build_monitor_collector_from_config(
        config=resolved_config,
        force_enable=(guard_mode == GuardMode.MONITOR),
    )
    structured_emitter = build_structured_emitter_from_config(config=resolved_config, logger_name="omega.api")
    telemetry_service = AnonymousTelemetryService(
        config=resolved_config,
        dispatcher=notification_dispatcher,
        surface="api",
    )
    incident_export_cfg = IncidentExportConfig.from_cfg(api_cfg.get("incident_export", {}) or {})
    incident_replay_cfg = IncidentReplayConfig.from_cfg(api_cfg.get("incident_replay", {}) or {})
    incident_export_store: Optional[IncidentExportStore] = None
    incident_api_key_store: Optional[IncidentApiKeyStore] = None
    incident_rate_limiter: Optional[IncidentRateLimiter] = None
    incident_replay_store: Optional[IncidentReplayStore] = None
    incident_replay_manager: Optional[IncidentReplayJobManager] = None
    if incident_export_cfg.enabled:
        incident_export_store = IncidentExportStore(
            sqlite_path=incident_export_cfg.store_path,
            retention_days=incident_export_cfg.retention_days,
        )
        incident_api_key_store = IncidentApiKeyStore(sqlite_path=incident_export_cfg.key_store_path)
        incident_rate_limiter = IncidentRateLimiter(
            rpm=incident_export_cfg.rate_limit_rpm,
            burst=incident_export_cfg.rate_limit_burst,
        )
    if incident_replay_cfg.enabled and incident_export_store is not None:
        incident_replay_store = IncidentReplayStore(sqlite_path=incident_replay_cfg.store_path)
        incident_replay_manager = IncidentReplayJobManager(
            config=incident_replay_cfg,
            replay_store=incident_replay_store,
            incident_store=incident_export_store,
            incident_retention_days=incident_export_cfg.retention_days,
        )
    return ScanRuntime(
        config=resolved_config,
        projector=build_projector(resolved_config),
        omega_core=OmegaCoreV1(omega_params_from_config(resolved_config)),
        off_policy=OffPolicyV1(resolved_config),
        api_keys=api_keys,
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
        notification_dispatcher=notification_dispatcher,
        monitor_collector=monitor_collector,
        structured_emitter=structured_emitter,
        incident_export_cfg=incident_export_cfg,
        incident_export_store=incident_export_store,
        incident_api_key_store=incident_api_key_store,
        incident_rate_limiter=incident_rate_limiter,
        incident_replay_cfg=incident_replay_cfg,
        incident_replay_store=incident_replay_store,
        incident_replay_manager=incident_replay_manager,
        telemetry_service=telemetry_service,
        license_status=(verified_license.to_dict() if verified_license is not None else None),
        skillbox=SkillBox.from_config(resolved_config),
    )


def build_runtime(resolved_config: Dict[str, Any]) -> ScanRuntime:
    return _make_runtime(resolved_config)


def _runtime_config(runtime: ScanRuntime) -> ApiRuntime:
    if isinstance(runtime.runtime_cfg, ApiRuntime):
        return runtime.runtime_cfg
    api_cfg = runtime.config.get("api", {}) if isinstance(runtime.config.get("api", {}), dict) else {}
    return ApiRuntime.from_cfg(api_cfg.get("runtime", {}) or {})


def _guard_mode(runtime: ScanRuntime) -> GuardMode:
    return resolve_guard_mode(runtime.config)


def _effective_runtime_mode(runtime: ScanRuntime, parsed: Mapping[str, Any]) -> str:
    runtime_cfg = _runtime_config(runtime)
    mode = str(runtime_cfg.mode)
    req_mode = str(parsed.get("runtime_mode", "") or "").strip().lower()
    if runtime_cfg.allow_request_override and req_mode in {"stateless", "stateful"}:
        mode = req_mode
    if mode == "stateful" and not str(parsed.get("session_id") or "").strip():
        raise HTTPException(status_code=400, detail="session_id_required_stateful")
    return mode
