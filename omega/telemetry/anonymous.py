"""Anonymous product/security telemetry runtime (OSS + Enterprise compatible)."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import sys
import threading
import time
from typing import Any, Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest
import uuid

import jsonschema

from omega.notifications.models import RiskEvent, new_event_id, utc_now_iso

LOGGER = logging.getLogger(__name__)


TELEMETRY_BATCH_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "telemetry_batch_v1.json",
    "title": "telemetry_batch_v1",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version",
        "batch_id",
        "instance",
        "window",
        "product_metrics",
        "security_analytics",
        "attack_signatures",
    ],
    "properties": {
        "schema_version": {"const": "telemetry_batch_v1"},
        "batch_id": {"type": "string", "minLength": 8},
        "instance": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "instance_id",
                "core_version",
                "tier",
                "deployment_mode",
                "os_arch",
                "python_version",
            ],
            "properties": {
                "instance_id": {"type": "string"},
                "core_version": {"type": "string"},
                "tier": {"type": "string", "enum": ["oss", "enterprise"]},
                "deployment_mode": {"type": "string", "enum": ["lib", "sidecar", "gateway", "unknown"]},
                "os_arch": {"type": "string"},
                "python_version": {"type": "string"},
            },
        },
        "window": {"type": "string", "enum": ["24h"]},
        "product_metrics": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "uptime_bucket",
                "modules_enabled",
                "fallback_events_count",
                "config_changes_count",
                "last_sync_status",
            ],
            "properties": {
                "uptime_bucket": {"type": "string", "enum": ["0-1", "1-6", "6-24", "24+"]},
                "modules_enabled": {"type": "array", "items": {"type": "string"}},
                "fallback_events_count": {"type": "integer", "minimum": 0},
                "config_changes_count": {"type": "integer", "minimum": 0},
                "last_sync_status": {"type": "string"},
            },
        },
        "security_analytics": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "attack_type_counts",
                "policy_triggers",
                "tool_abuse_categories",
                "risk_score_buckets",
                "fp_reports_count",
                "enforcement_actions",
            ],
            "properties": {
                "attack_type_counts": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
                "policy_triggers": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
                "tool_abuse_categories": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
                "risk_score_buckets": {"type": "array", "items": {"type": "number"}},
                "fp_reports_count": {"type": "integer", "minimum": 0},
                "enforcement_actions": {"type": "object", "additionalProperties": {"type": "integer", "minimum": 0}},
            },
        },
        "attack_signatures": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["pattern_hash", "rule_id", "accumulation_steps", "provenance_type", "count"],
                "properties": {
                    "pattern_hash": {"type": "string", "minLength": 32},
                    "rule_id": {"type": "string"},
                    "accumulation_steps": {"type": "integer", "minimum": 0},
                    "provenance_type": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                },
            },
        },
    },
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _non_empty_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return ""


def _parse_bool_text(raw: str) -> Optional[bool]:
    txt = str(raw or "").strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return None


def _safe_json_dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _runtime_core_version() -> str:
    try:
        from importlib import metadata as importlib_metadata

        return str(importlib_metadata.version("omega-walls"))
    except Exception:  # noqa: BLE001
        return "unknown"


def _normalize_attack_type(wall_name: str) -> str:
    mapping = {
        "override_instructions": "prompt_injection",
        "policy_evasion": "policy_drift",
        "tool_or_action_abuse": "tool_abuse",
        "secret_exfiltration": "data_exfil",
    }
    return mapping.get(str(wall_name).strip().lower(), "unknown")


def _bucket_risk(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 1)


def _uptime_bucket(hours: float) -> str:
    h = float(max(0.0, hours))
    if h < 1.0:
        return "0-1"
    if h < 6.0:
        return "1-6"
    if h < 24.0:
        return "6-24"
    return "24+"


def _normalize_deployment_mode(raw: str) -> str:
    mode = str(raw or "").strip().lower()
    if mode in {"lib", "sidecar", "gateway"}:
        return mode
    return "unknown"


@dataclass
class TelemetryConfig:
    enabled: bool = True
    endpoint: str = "https://telemetry.omega-walls.io/v1/collect"
    interval_hours: int = 24
    max_batch_kb: int = 50
    retry_schedule_sec: List[int] = field(default_factory=lambda: [60, 300, 900])
    tier: str = "oss"
    deployment_mode: str = "auto"
    policy_privacy_url: str = "https://github.com/synqratech/omega-walls/tree/main/docs#privacy"
    policy_dpa_url: str = "https://github.com/synqratech/omega-walls/tree/main/docs#data-processing"
    audit_log_path: str = "artifacts/logs/telemetry_audit.log"
    state_path: str = "artifacts/state/telemetry_state.json"
    queue_max_events: int = 2048
    send_timeout_sec: float = 8.0
    env_override_enabled: Optional[bool] = None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TelemetryConfig":
        raw = config.get("telemetry", {}) if isinstance(config.get("telemetry", {}), Mapping) else {}
        policy_urls = raw.get("policy_urls", {}) if isinstance(raw.get("policy_urls", {}), Mapping) else {}
        env_raw = os.environ.get("OMEGA_TELEMETRY", "")
        env_override = _parse_bool_text(env_raw) if str(env_raw).strip() else None
        enabled_cfg = bool(raw.get("enabled", True))
        enabled = bool(env_override) if env_override is not None else enabled_cfg
        return cls(
            enabled=enabled,
            endpoint=str(raw.get("endpoint", "https://telemetry.omega-walls.io/v1/collect")).strip()
            or "https://telemetry.omega-walls.io/v1/collect",
            interval_hours=max(1, int(raw.get("interval_hours", 24))),
            max_batch_kb=max(1, int(raw.get("max_batch_kb", 50))),
            retry_schedule_sec=[max(1, int(x)) for x in list(raw.get("retry_schedule_sec", [60, 300, 900]))] or [60, 300, 900],
            tier=(str(raw.get("tier", "oss")).strip().lower() or "oss"),
            deployment_mode=(str(raw.get("deployment_mode", "auto")).strip().lower() or "auto"),
            policy_privacy_url=str(policy_urls.get("privacy", "")).strip()
            or "https://github.com/synqratech/omega-walls/tree/main/docs#privacy",
            policy_dpa_url=str(policy_urls.get("dpa", "")).strip()
            or "https://github.com/synqratech/omega-walls/tree/main/docs#data-processing",
            audit_log_path=str(raw.get("audit_log_path", "artifacts/logs/telemetry_audit.log")).strip()
            or "artifacts/logs/telemetry_audit.log",
            state_path=str(raw.get("state_path", "artifacts/state/telemetry_state.json")).strip()
            or "artifacts/state/telemetry_state.json",
            env_override_enabled=env_override,
        )


class TelemetryStateStore:
    def __init__(self, *, path: str | Path) -> None:
        self.path = Path(str(path))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._state = self._load()

    def _default(self) -> Dict[str, Any]:
        return {
            "instance_id": str(uuid.uuid4()),
            "first_seen_at": _utc_now_iso(),
            "override_enabled": None,
            "startup_notified": False,
            "next_send_at": None,
            "last_send_status": {},
        }

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            state = self._default()
            self._write(state)
            return state
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("invalid_state_payload")
        except Exception:  # noqa: BLE001
            payload = self._default()
            self._write(payload)
            return payload
        state = self._default()
        state.update(payload)
        if not str(state.get("instance_id", "")).strip():
            state["instance_id"] = str(uuid.uuid4())
        return state

    def _write(self, state: Mapping[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(dict(state), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def update(self, patch: Mapping[str, Any]) -> Dict[str, Any]:
        with self._lock:
            self._state.update(dict(patch or {}))
            self._write(self._state)
            return dict(self._state)


class CollectionQueue:
    def __init__(self, *, max_events: int) -> None:
        self._buf: Deque[Dict[str, Any]] = deque(maxlen=max(8, int(max_events)))
        self._lock = threading.Lock()

    def append(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            self._buf.append(dict(event))

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def size(self) -> int:
        with self._lock:
            return int(len(self._buf))


class AllowlistSanitizer:
    _DENY_KEYS = {
        "prompt",
        "raw_prompt",
        "raw_document",
        "document_text",
        "content",
        "payload",
        "payload_raw",
        "api_key",
        "api_keys",
        "token",
        "tokens",
        "password",
        "secret",
        "authorization",
        "cookie",
        "set-cookie",
        "ip",
        "ip_address",
        "hostname",
        "host",
        "mac",
        "dns",
        "internal_path",
        "path",
    }

    _ALLOWED_TOP_KEYS = {
        "surface",
        "control_outcome",
        "severity",
        "attack_types",
        "policy_triggers",
        "enforcement_actions",
        "risk_score",
        "accumulation_steps",
        "fallback_active",
        "fallback_level",
        "modules_enabled",
        "sync_status",
        "pattern_parts",
        "rule_id",
        "provenance_type",
        "fp_reported",
    }

    @classmethod
    def _strip_recursive(cls, value: Any) -> Any:
        if isinstance(value, Mapping):
            out: Dict[str, Any] = {}
            for key, inner in dict(value).items():
                key_norm = str(key).strip().lower()
                if key_norm in cls._DENY_KEYS:
                    continue
                out[str(key)] = cls._strip_recursive(inner)
            return out
        if isinstance(value, list):
            return [cls._strip_recursive(x) for x in list(value)]
        return value

    @classmethod
    def sanitize(cls, raw: Mapping[str, Any]) -> Dict[str, Any]:
        stripped = cls._strip_recursive(raw)
        out: Dict[str, Any] = {}
        for key in cls._ALLOWED_TOP_KEYS:
            if key in stripped:
                out[key] = stripped[key]
        attack_types = out.get("attack_types", [])
        policy_triggers = out.get("policy_triggers", [])
        enforcement_actions = out.get("enforcement_actions", [])
        modules_enabled = out.get("modules_enabled", [])
        pattern_parts = out.get("pattern_parts", [])
        out["surface"] = str(out.get("surface", "runtime")).strip().lower() or "runtime"
        out["control_outcome"] = str(out.get("control_outcome", "ALLOW")).strip().upper() or "ALLOW"
        out["severity"] = str(out.get("severity", "L1")).strip().upper() or "L1"
        out["attack_types"] = sorted({str(x).strip().lower() for x in list(attack_types) if str(x).strip()})
        out["policy_triggers"] = sorted({str(x).strip().lower() for x in list(policy_triggers) if str(x).strip()})
        out["enforcement_actions"] = sorted({str(x).strip().upper() for x in list(enforcement_actions) if str(x).strip()})
        out["risk_score"] = float(max(0.0, min(1.0, float(out.get("risk_score", 0.0)))))
        out["accumulation_steps"] = max(0, int(out.get("accumulation_steps", 0)))
        out["fallback_active"] = bool(out.get("fallback_active", False))
        out["fallback_level"] = str(out.get("fallback_level", "none")).strip().lower() or "none"
        out["modules_enabled"] = sorted({str(x).strip().lower() for x in list(modules_enabled) if str(x).strip()})
        out["sync_status"] = str(out.get("sync_status", "ok")).strip().lower() or "ok"
        out["pattern_parts"] = [str(x).strip().lower() for x in list(pattern_parts) if str(x).strip()][:12]
        out["rule_id"] = str(out.get("rule_id", "unknown")).strip().lower() or "unknown"
        out["provenance_type"] = str(out.get("provenance_type", "unknown")).strip().lower() or "unknown"
        out["fp_reported"] = bool(out.get("fp_reported", False))
        return out


@dataclass
class _AggregateState:
    events_total: int = 0
    attack_type_counts: Counter = field(default_factory=Counter)
    policy_triggers: Counter = field(default_factory=Counter)
    tool_abuse_categories: Counter = field(default_factory=Counter)
    risk_score_buckets: Counter = field(default_factory=Counter)
    enforcement_actions: Counter = field(default_factory=Counter)
    fallback_events_count: int = 0
    fp_reports_count: int = 0
    config_changes_count: int = 0
    modules_enabled: set = field(default_factory=set)
    last_sync_status: str = "ok"
    attack_signatures: Counter = field(default_factory=Counter)


class DailyAggregator:
    def __init__(self) -> None:
        self._state = _AggregateState()
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._state = _AggregateState()

    def add(self, event: Mapping[str, Any]) -> None:
        with self._lock:
            st = self._state
            st.events_total += 1
            for item in list(event.get("attack_types", []) or []):
                at = str(item).strip().lower()
                if not at:
                    continue
                st.attack_type_counts[at] += 1
                if at == "tool_abuse":
                    for act in list(event.get("enforcement_actions", []) or []):
                        st.tool_abuse_categories[str(act).strip().lower()] += 1
            for item in list(event.get("policy_triggers", []) or []):
                val = str(item).strip().lower()
                if val:
                    st.policy_triggers[val] += 1
            for action in list(event.get("enforcement_actions", []) or []):
                val = str(action).strip().upper()
                if val:
                    st.enforcement_actions[val] += 1
            rb = _bucket_risk(float(event.get("risk_score", 0.0)))
            st.risk_score_buckets[str(rb)] += 1
            if bool(event.get("fallback_active", False)):
                st.fallback_events_count += 1
            if bool(event.get("fp_reported", False)):
                st.fp_reports_count += 1
            for mod in list(event.get("modules_enabled", []) or []):
                m = str(mod).strip().lower()
                if m:
                    st.modules_enabled.add(m)
            st.last_sync_status = str(event.get("sync_status", st.last_sync_status)).strip().lower() or st.last_sync_status
            pattern_parts = [str(x).strip().lower() for x in list(event.get("pattern_parts", []) or []) if str(x).strip()]
            if pattern_parts:
                pattern_hash = _sha256_hex(" -> ".join(pattern_parts))
                signature_key = json.dumps(
                    {
                        "pattern_hash": pattern_hash,
                        "rule_id": str(event.get("rule_id", "unknown")).strip().lower() or "unknown",
                        "accumulation_steps": int(event.get("accumulation_steps", 0)),
                        "provenance_type": str(event.get("provenance_type", "unknown")).strip().lower() or "unknown",
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                )
                st.attack_signatures[signature_key] += 1

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            st = self._state
            signatures: List[Dict[str, Any]] = []
            for key, count in st.attack_signatures.items():
                try:
                    row = dict(json.loads(key))
                    row["count"] = int(count)
                    signatures.append(row)
                except Exception:  # noqa: BLE001
                    continue
            signatures.sort(key=lambda x: (-int(x.get("count", 0)), str(x.get("pattern_hash", ""))))
            risk_values: List[float] = []
            for bucket, count in sorted(st.risk_score_buckets.items(), key=lambda kv: float(kv[0])):
                risk_values.extend([float(bucket)] * int(count))
            return {
                "events_total": int(st.events_total),
                "attack_type_counts": {str(k): int(v) for k, v in st.attack_type_counts.items()},
                "policy_triggers": {str(k): int(v) for k, v in st.policy_triggers.items()},
                "tool_abuse_categories": {str(k): int(v) for k, v in st.tool_abuse_categories.items()},
                "risk_score_buckets": list(risk_values),
                "enforcement_actions": {str(k): int(v) for k, v in st.enforcement_actions.items()},
                "fallback_events_count": int(st.fallback_events_count),
                "fp_reports_count": int(st.fp_reports_count),
                "config_changes_count": int(st.config_changes_count),
                "modules_enabled": sorted({str(x) for x in st.modules_enabled}),
                "last_sync_status": str(st.last_sync_status),
                "attack_signatures": signatures,
            }


class TelemetrySender:
    def __init__(self, *, endpoint: str, timeout_sec: float = 8.0) -> None:
        self.endpoint = str(endpoint)
        self.timeout_sec = float(timeout_sec)

    def post(self, payload: Mapping[str, Any]) -> Tuple[int, str]:
        raw = _safe_json_dump(payload).encode("utf-8")
        body = gzip.compress(raw, compresslevel=6)
        req = urlrequest.Request(
            self.endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=self.timeout_sec) as resp:  # noqa: S310
                code = int(getattr(resp, "status", 202))
                payload_raw = resp.read().decode("utf-8", errors="replace")
                return code, payload_raw
        except urlerror.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            return int(exc.code), body_text


class AnonymousTelemetryService:
    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        dispatcher: Optional[Any] = None,
        surface: str,
        start_worker: bool = True,
        emit_startup_notice: bool = True,
    ) -> None:
        self.runtime_config = dict(config or {})
        self.config = TelemetryConfig.from_config(self.runtime_config)
        self.surface = str(surface or "runtime").strip().lower() or "runtime"
        self.dispatcher = dispatcher
        self._state_store = TelemetryStateStore(path=self.config.state_path)
        self._queue = CollectionQueue(max_events=self.config.queue_max_events)
        self._agg = DailyAggregator()
        self._sender = TelemetrySender(endpoint=self.config.endpoint, timeout_sec=self.config.send_timeout_sec)
        self._lock = threading.Lock()
        self._started_at = _utc_now()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._enabled, self._effective_source = self._resolve_enabled_state()
        self._ensure_next_send_time()
        if self._enabled and bool(emit_startup_notice):
            self._startup_notice_once()
        if bool(start_worker):
            self._worker = threading.Thread(target=self._worker_loop, daemon=True, name=f"omega-telemetry-{self.surface}")
            self._worker.start()

    def _resolve_enabled_state(self) -> Tuple[bool, str]:
        st = self._state_store.snapshot()
        override_enabled = st.get("override_enabled", None)
        if isinstance(override_enabled, bool):
            return bool(override_enabled), "override"
        if self.config.env_override_enabled is not None:
            return bool(self.config.env_override_enabled), "env"
        return bool(self.config.enabled), "config"

    def _ensure_next_send_time(self) -> None:
        st = self._state_store.snapshot()
        if _non_empty_text(st.get("next_send_at", "")):
            return
        due = (_utc_now() + timedelta(hours=int(self.config.interval_hours))).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._state_store.update({"next_send_at": due})

    def _telemetry_notice_text(self) -> str:
        return (
            "Omega Walls telemetry is enabled. Anonymous usage and attack pattern data helps improve security. "
            "No content, keys, or PII is collected. Disable: OMEGA_TELEMETRY=false or telemetry.enabled=false. "
            f"Privacy: {self.config.policy_privacy_url} DPA: {self.config.policy_dpa_url}"
        )

    def _startup_notice_once(self) -> None:
        st = self._state_store.snapshot()
        if bool(st.get("startup_notified", False)):
            return
        text = self._telemetry_notice_text()
        if bool(getattr(sys.stdout, "isatty", lambda: False)()):
            print(f"INFO: {text}", flush=True)  # noqa: T201
        LOGGER.info("telemetry_startup_notice: %s", text)
        notifications_cfg = self.config_from_runtime().get("notifications", {})
        notifications_enabled = bool(notifications_cfg.get("enabled", False)) if isinstance(notifications_cfg, Mapping) else False
        if self.dispatcher is not None and notifications_enabled:
            event = RiskEvent(
                event_id=new_event_id(),
                timestamp=utc_now_iso(),
                surface=self.surface,
                control_outcome="ALLOW",
                triggers=["STARTUP_TELEMETRY"],
                reasons=[],
                action_types=[],
                trace_id=f"startup-trace-{uuid.uuid4().hex[:16]}",
                decision_id=f"startup-decision-{uuid.uuid4().hex[:16]}",
                payload_redacted={"event_kind": "startup_telemetry", "startup_text": text},
                event_kind="startup_telemetry",
            )
            try:
                self.dispatcher.emit_startup_event(event, startup_kind="telemetry", once_per_process=True)
            except Exception:  # noqa: BLE001
                LOGGER.warning("telemetry startup webhook notification failed", exc_info=True)
        self._state_store.update({"startup_notified": True})

    def config_from_runtime(self) -> Mapping[str, Any]:
        return self.runtime_config

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def emit_event(self, raw_event: Mapping[str, Any]) -> None:
        if not self._enabled:
            return
        try:
            event = AllowlistSanitizer.sanitize(raw_event)
            self._queue.append(event)
            self._agg.add(event)
        except Exception:  # noqa: BLE001
            LOGGER.warning("telemetry emit_event failed (fail-open)", exc_info=True)

    def disable(self) -> Dict[str, Any]:
        with self._lock:
            self._enabled = False
            self._effective_source = "override"
            self._queue.clear()
            self._agg.clear()
            st = self._state_store.update(
                {
                    "override_enabled": False,
                    "instance_id": str(uuid.uuid4()),
                    "startup_notified": False,
                    "next_send_at": (_utc_now() + timedelta(hours=int(self.config.interval_hours))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "last_send_status": {"status": "disabled", "ts": _utc_now_iso()},
                }
            )
            return {"status": "ok", "enabled": False, "instance_id": str(st.get("instance_id", ""))}

    def _build_payload(self) -> Dict[str, Any]:
        st = self._state_store.snapshot()
        agg = self._agg.snapshot()
        deployment_mode = _normalize_deployment_mode(self.config.deployment_mode if self.config.deployment_mode != "auto" else self.surface)
        uptime_hours = (_utc_now() - self._started_at).total_seconds() / 3600.0
        payload = {
            "schema_version": "telemetry_batch_v1",
            "batch_id": str(uuid.uuid4()),
            "instance": {
                "instance_id": str(st.get("instance_id", "")),
                "core_version": _runtime_core_version(),
                "tier": ("enterprise" if str(self.config.tier).lower() == "enterprise" else "oss"),
                "deployment_mode": deployment_mode,
                "os_arch": f"{platform.system().lower()}-{platform.machine().lower()}",
                "python_version": str(platform.python_version()),
            },
            "window": "24h",
            "product_metrics": {
                "uptime_bucket": _uptime_bucket(uptime_hours),
                "modules_enabled": list(agg.get("modules_enabled", [])),
                "fallback_events_count": int(agg.get("fallback_events_count", 0)),
                "config_changes_count": int(agg.get("config_changes_count", 0)),
                "last_sync_status": str(agg.get("last_sync_status", "ok")),
            },
            "security_analytics": {
                "attack_type_counts": dict(agg.get("attack_type_counts", {})),
                "policy_triggers": dict(agg.get("policy_triggers", {})),
                "tool_abuse_categories": dict(agg.get("tool_abuse_categories", {})),
                "risk_score_buckets": list(agg.get("risk_score_buckets", [])),
                "fp_reports_count": int(agg.get("fp_reports_count", 0)),
                "enforcement_actions": dict(agg.get("enforcement_actions", {})),
            },
            "attack_signatures": list(agg.get("attack_signatures", [])),
        }
        max_bytes = int(self.config.max_batch_kb) * 1024
        while True:
            raw = _safe_json_dump(payload).encode("utf-8")
            if len(raw) <= max_bytes:
                break
            signs = list(payload.get("attack_signatures", []))
            if not signs:
                break
            payload["attack_signatures"] = signs[:-1]
        return payload

    def _validate_payload(self, payload: Mapping[str, Any]) -> None:
        jsonschema.validate(instance=dict(payload), schema=TELEMETRY_BATCH_SCHEMA)

    def _write_audit(self, *, payload: Mapping[str, Any], status: str, detail: str, attempts: int) -> None:
        audit_path = Path(str(self.config.audit_log_path))
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        payload_text = _safe_json_dump(payload)
        row = {
            "ts": _utc_now_iso(),
            "payload_hash": _sha256_hex(payload_text),
            "payload_size": len(payload_text.encode("utf-8")),
            "status": str(status),
            "detail": str(detail),
            "attempts": int(attempts),
        }
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    def _drain_with_send(self) -> None:
        payload = self._build_payload()
        attempts = 0
        try:
            self._validate_payload(payload)
        except Exception as exc:  # noqa: BLE001
            self._write_audit(payload=payload, status="schema_invalid_drop", detail=str(exc), attempts=0)
            self._agg.clear()
            self._queue.clear()
            self._state_store.update(
                {
                    "next_send_at": (_utc_now() + timedelta(hours=int(self.config.interval_hours))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "last_send_status": {"status": "schema_invalid_drop", "detail": str(exc), "ts": _utc_now_iso()},
                }
            )
            return
        schedules = [0] + list(self.config.retry_schedule_sec)
        final_status = "send_failed_drop"
        final_detail = "transport_error"
        for delay_sec in schedules:
            if delay_sec > 0:
                time.sleep(float(delay_sec))
            attempts += 1
            try:
                code, body = self._sender.post(payload)
            except Exception as exc:  # noqa: BLE001
                final_status = "send_failed_drop"
                final_detail = str(exc)
                continue
            if code in {200, 201, 202, 204}:
                final_status = "accepted"
                final_detail = f"http_{code}"
                break
            if code == 409:
                final_status = "duplicate"
                final_detail = "http_409"
                break
            if code == 400:
                final_status = "schema_error_drop"
                final_detail = body[:256]
                break
            if code >= 500 or code == 429:
                final_status = "send_failed_drop"
                final_detail = f"http_{code}"
                continue
            final_status = "http_error_drop"
            final_detail = f"http_{code}:{body[:128]}"
            break
        self._write_audit(payload=payload, status=final_status, detail=final_detail, attempts=attempts)
        self._agg.clear()
        self._queue.clear()
        self._state_store.update(
            {
                "next_send_at": (_utc_now() + timedelta(hours=int(self.config.interval_hours))).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "last_send_status": {"status": final_status, "detail": final_detail, "attempts": attempts, "ts": _utc_now_iso()},
            }
        )

    def _next_due(self) -> datetime:
        st = self._state_store.snapshot()
        raw = _non_empty_text(st.get("next_send_at", ""))
        if not raw:
            return _utc_now()
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return _utc_now()

    def _pending_payload_size(self) -> int:
        payload = self._build_payload()
        return len(_safe_json_dump(payload).encode("utf-8"))

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self._enabled:
                    time.sleep(0.5)
                    continue
                has_pending = self._queue.size() > 0
                if not has_pending:
                    time.sleep(0.5)
                    continue
                due = _utc_now() >= self._next_due()
                size_hit = self._pending_payload_size() >= int(self.config.max_batch_kb) * 1024
                if due or size_hit:
                    self._drain_with_send()
                else:
                    time.sleep(0.5)
            except Exception:  # noqa: BLE001
                LOGGER.warning("telemetry worker loop error", exc_info=True)
                time.sleep(0.5)

    def status_snapshot(self) -> Dict[str, Any]:
        st = self._state_store.snapshot()
        return {
            "enabled": bool(self._enabled),
            "effective_source": str(self._effective_source),
            "next_send_at": st.get("next_send_at"),
            "pending_events": int(self._queue.size()),
            "pending_batch_bytes": int(self._pending_payload_size()) if self._queue.size() > 0 else 0,
            "last_send_status": dict(st.get("last_send_status", {}) or {}),
            "instance_id": str(st.get("instance_id", "")),
            "tier": str(self.config.tier),
            "endpoint": str(self.config.endpoint),
        }

    def show_pending(self) -> Dict[str, Any]:
        if not self._enabled:
            return {"enabled": False, "payload": None}
        payload = self._build_payload()
        self._validate_payload(payload)
        return {"enabled": True, "payload": payload}


def build_telemetry_event(
    *,
    surface: str,
    control_outcome: str,
    severity: str,
    walls_triggered: Sequence[str],
    reason_codes: Sequence[str],
    action_types: Sequence[str],
    risk_score: float,
    fallback_active: bool,
    fallback_level: str,
    accumulation_steps: int,
    provenance_type: str,
    module_flags: Mapping[str, bool],
    fp_reported: bool = False,
) -> Dict[str, Any]:
    attack_types = [_normalize_attack_type(x) for x in list(walls_triggered)]
    attack_types = [x for x in attack_types if str(x).strip()]
    pattern_parts = [f"src:{str(provenance_type).strip().lower() or 'unknown'}"]
    pattern_parts.extend(sorted({str(x).strip().lower() for x in attack_types if str(x).strip()}))
    pattern_parts.append(f"outcome:{str(control_outcome).strip().lower() or 'allow'}")
    if bool(fallback_active):
        pattern_parts.append(f"fallback:{str(fallback_level).strip().lower() or 'unknown'}")
    rule_id = str(list(walls_triggered)[0]).strip().lower() if list(walls_triggered) else "none"
    modules_enabled = sorted([k for k, v in dict(module_flags or {}).items() if bool(v)])
    return AllowlistSanitizer.sanitize(
        {
            "surface": str(surface),
            "control_outcome": str(control_outcome),
            "severity": str(severity),
            "attack_types": attack_types,
            "policy_triggers": list(reason_codes),
            "enforcement_actions": list(action_types),
            "risk_score": float(risk_score),
            "accumulation_steps": int(accumulation_steps),
            "fallback_active": bool(fallback_active),
            "fallback_level": str(fallback_level),
            "modules_enabled": modules_enabled,
            "sync_status": "ok",
            "pattern_parts": pattern_parts,
            "rule_id": rule_id,
            "provenance_type": str(provenance_type),
            "fp_reported": bool(fp_reported),
        }
    )
