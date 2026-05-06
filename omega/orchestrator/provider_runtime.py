from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from urllib import error as urlerror
from urllib import request as urlrequest

from hashlib import sha256

LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def mask_secret(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    tail = text[-4:] if len(text) >= 4 else text
    return f"sk-...{tail}"


def _b64_key_material_from_env_or_keyring(*, env_name: str = "OMEGA_MASTER_KEY") -> str:
    env_val = str(os.environ.get(env_name, "")).strip()
    if env_val:
        return env_val
    try:
        import keyring  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing_master_key_env:{env_name}") from exc
    service_name = "omega-walls"
    user_name = "orchestrator-master-key"
    stored = keyring.get_password(service_name, user_name)
    if stored:
        return str(stored).strip()
    generated = f"hex:{secrets.token_hex(32)}"
    try:
        keyring.set_password(service_name, user_name, generated)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"missing_master_key_env:{env_name}") from exc
    return generated


class _VaultCrypto:
    def __init__(self, *, key_material: str) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        km = str(key_material or "").strip()
        if not km:
            raise ValueError("missing_master_key")
        if km.startswith("base64:"):
            import base64

            key_bytes = base64.b64decode(km.split(":", 1)[1].encode("ascii"))
        elif km.startswith("hex:"):
            key_bytes = bytes.fromhex(km.split(":", 1)[1].strip())
        else:
            key_bytes = sha256(km.encode("utf-8")).digest()
        if len(key_bytes) < 32:
            key_bytes = key_bytes.ljust(32, b"\0")
        self._key = key_bytes[:32]
        self._aesgcm = AESGCM(self._key)

    def encrypt(self, payload: bytes) -> bytes:
        nonce = secrets.token_bytes(12)
        encrypted = self._aesgcm.encrypt(nonce, payload, None)
        return nonce + encrypted

    def decrypt(self, payload: bytes) -> bytes:
        if len(payload) < 13:
            raise ValueError("invalid_encrypted_payload")
        nonce = payload[:12]
        data = payload[12:]
        return self._aesgcm.decrypt(nonce, data, None)


@dataclass(frozen=True)
class ProviderCandidate:
    provider_id: str
    provider_type: str
    model: str
    base_url: str
    key_slot: str
    key_ref: str
    priority: int


@dataclass(frozen=True)
class OrchestratorAlertEvent:
    alert_type: str
    provider_id: str
    message: str
    protection_mode: str
    reason: str
    ts: str
    details: Dict[str, Any] = field(default_factory=dict)


class ProviderHealthState:
    HEALTHY = "healthy"
    WARNING = "warning"
    EXHAUSTED = "exhausted"
    FALLBACK_ACTIVE = "fallback_active"
    RECOVERING = "recovering"


@dataclass
class OrchestratorConfig:
    enabled: bool = False
    sqlite_path: str = "artifacts/state/provider_orchestrator.db"
    master_key_env: str = "OMEGA_MASTER_KEY"
    fallback_mode: str = "rule_only"
    threshold_errors: int = 3
    threshold_window_sec: int = 60
    healthcheck_interval_sec: int = 180
    alerts_cooldown_sec: int = 900
    providers: List[ProviderCandidate] = field(default_factory=list)

    @classmethod
    def from_api_cfg(cls, api_cfg: Mapping[str, Any], *, default_provider: str, default_model: str, default_base_url: str) -> "OrchestratorConfig":
        raw = api_cfg.get("orchestrator", {}) if isinstance(api_cfg.get("orchestrator", {}), Mapping) else {}
        enabled = bool(raw.get("enabled", False))
        store_cfg = raw.get("store", {}) if isinstance(raw.get("store", {}), Mapping) else {}
        fallback_cfg = raw.get("fallback", {}) if isinstance(raw.get("fallback", {}), Mapping) else {}
        threshold_cfg = fallback_cfg.get("threshold", {}) if isinstance(fallback_cfg.get("threshold", {}), Mapping) else {}
        recovery_cfg = raw.get("recovery", {}) if isinstance(raw.get("recovery", {}), Mapping) else {}
        alerts_cfg = raw.get("alerts", {}) if isinstance(raw.get("alerts", {}), Mapping) else {}
        rows = raw.get("providers", []) if isinstance(raw.get("providers", []), list) else []
        providers: List[ProviderCandidate] = []
        for idx, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            provider_id = str(row.get("id", "")).strip()
            provider_type = str(row.get("type", "")).strip().lower()
            if not provider_id or not provider_type:
                continue
            providers.append(
                ProviderCandidate(
                    provider_id=provider_id,
                    provider_type=provider_type,
                    model=str(row.get("model", default_model)).strip() or default_model,
                    base_url=str(row.get("base_url", default_base_url)).strip() or default_base_url,
                    key_slot=str(row.get("key_slot", "primary")).strip().lower() or "primary",
                    key_ref=str(row.get("primary_ref", provider_id)).strip() or provider_id,
                    priority=int(row.get("priority", idx)),
                )
            )
            backup_ref = str(row.get("backup_ref", "")).strip()
            if backup_ref:
                providers.append(
                    ProviderCandidate(
                        provider_id=provider_id,
                        provider_type=provider_type,
                        model=str(row.get("model", default_model)).strip() or default_model,
                        base_url=str(row.get("base_url", default_base_url)).strip() or default_base_url,
                        key_slot="backup",
                        key_ref=backup_ref,
                        priority=int(row.get("priority", idx)) + 1,
                    )
                )
        if not providers:
            providers = [
                ProviderCandidate(
                    provider_id="primary",
                    provider_type=str(default_provider).strip().lower(),
                    model=str(default_model).strip(),
                    base_url=str(default_base_url).strip(),
                    key_slot="primary",
                    key_ref="primary",
                    priority=0,
                ),
                ProviderCandidate(
                    provider_id="primary",
                    provider_type=str(default_provider).strip().lower(),
                    model=str(default_model).strip(),
                    base_url=str(default_base_url).strip(),
                    key_slot="backup",
                    key_ref="primary",
                    priority=1,
                ),
            ]
        return cls(
            enabled=enabled,
            sqlite_path=str(store_cfg.get("sqlite_path", "artifacts/state/provider_orchestrator.db")).strip()
            or "artifacts/state/provider_orchestrator.db",
            master_key_env=str(raw.get("master_key_env", "OMEGA_MASTER_KEY")).strip() or "OMEGA_MASTER_KEY",
            fallback_mode=str(fallback_cfg.get("mode", "rule_only")).strip().lower() or "rule_only",
            threshold_errors=max(1, int(threshold_cfg.get("errors", 3))),
            threshold_window_sec=max(1, int(threshold_cfg.get("window_sec", 60))),
            healthcheck_interval_sec=max(120, min(300, int(recovery_cfg.get("healthcheck_interval_sec", 180)))),
            alerts_cooldown_sec=max(30, int(alerts_cfg.get("cooldown_sec", 900))),
            providers=sorted(providers, key=lambda x: (int(x.priority), str(x.provider_id), str(x.key_slot))),
        )


class ProviderKeyVault:
    def __init__(self, *, sqlite_path: str | Path, master_key_env: str = "OMEGA_MASTER_KEY") -> None:
        self.sqlite_path = Path(str(sqlite_path))
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        key_material = _b64_key_material_from_env_or_keyring(env_name=str(master_key_env))
        self.crypto = _VaultCrypto(key_material=key_material)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.sqlite_path), timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS provider_keys (
                  provider_id TEXT NOT NULL,
                  key_slot TEXT NOT NULL,
                  key_ref TEXT NOT NULL,
                  key_cipher BLOB NOT NULL,
                  key_mask TEXT NOT NULL,
                  status TEXT NOT NULL,
                  deprecated_until TEXT,
                  updated_at TEXT NOT NULL,
                  last_error TEXT,
                  PRIMARY KEY(provider_id, key_slot)
                );
                CREATE TABLE IF NOT EXISTS provider_state (
                  provider_id TEXT PRIMARY KEY,
                  health_state TEXT NOT NULL,
                  consecutive_quota_errors INTEGER NOT NULL,
                  window_started_at TEXT,
                  fallback_level TEXT NOT NULL,
                  fallback_reason TEXT,
                  active_slot TEXT,
                  last_error TEXT,
                  last_transition_at TEXT NOT NULL,
                  next_recovery_check_at TEXT,
                  quota_signal TEXT
                );
                CREATE TABLE IF NOT EXISTS orchestrator_settings (
                  setting_key TEXT PRIMARY KEY,
                  setting_value TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT NOT NULL,
                  actor TEXT NOT NULL,
                  event_type TEXT NOT NULL,
                  state_diff_json TEXT NOT NULL,
                  reason TEXT
                );
                """
            )

    def put_key(self, *, provider_id: str, slot: str, key_ref: str, key_plain: str, status: str = "active") -> None:
        key = str(key_plain or "").strip()
        if not key:
            raise ValueError("empty_key")
        enc = self.crypto.encrypt(key.encode("utf-8"))
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO provider_keys(provider_id, key_slot, key_ref, key_cipher, key_mask, status, deprecated_until, updated_at, last_error)
                VALUES(?, ?, ?, ?, ?, ?, NULL, ?, NULL)
                ON CONFLICT(provider_id, key_slot) DO UPDATE SET
                  key_ref=excluded.key_ref,
                  key_cipher=excluded.key_cipher,
                  key_mask=excluded.key_mask,
                  status=excluded.status,
                  deprecated_until=NULL,
                  updated_at=excluded.updated_at,
                  last_error=NULL
                """,
                (
                    str(provider_id),
                    str(slot),
                    str(key_ref),
                    enc,
                    mask_secret(key),
                    str(status),
                    now,
                ),
            )

    def rotate_primary(self, *, provider_id: str, key_ref: str, key_plain: str, grace_hours: int = 24) -> None:
        now = _utc_now()
        dep_until = (now + timedelta(hours=max(1, int(grace_hours)))).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE provider_keys
                   SET status='deprecated', deprecated_until=?, updated_at=?
                 WHERE provider_id=? AND key_slot='primary' AND status='active'
                """,
                (dep_until, _utc_now_iso(), str(provider_id)),
            )
        self.put_key(provider_id=provider_id, slot="primary", key_ref=key_ref, key_plain=key_plain, status="active")

    def get_key(self, *, provider_id: str, slot: str) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT key_cipher, status, deprecated_until FROM provider_keys WHERE provider_id=? AND key_slot=?",
                (str(provider_id), str(slot)),
            ).fetchone()
        if row is None:
            return None
        status = str(row["status"])
        dep_until = str(row["deprecated_until"] or "").strip()
        if status == "deprecated" and dep_until:
            try:
                if _utc_now() > datetime.fromisoformat(dep_until.replace("Z", "+00:00")):
                    return None
            except Exception:  # noqa: BLE001
                return None
        payload = bytes(row["key_cipher"])
        return self.crypto.decrypt(payload).decode("utf-8", errors="strict")

    def list_keys(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT provider_id, key_slot, key_ref, key_mask, status, deprecated_until, updated_at, last_error
                FROM provider_keys ORDER BY provider_id ASC, key_slot ASC
                """
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "provider_id": str(row["provider_id"]),
                    "slot": str(row["key_slot"]),
                    "key_ref": str(row["key_ref"]),
                    "key_mask": str(row["key_mask"]),
                    "status": str(row["status"]),
                    "deprecated_until": (str(row["deprecated_until"]) if row["deprecated_until"] else None),
                    "updated_at": str(row["updated_at"]),
                    "last_error": (str(row["last_error"]) if row["last_error"] else None),
                }
            )
        return out

    def set_key_error(self, *, provider_id: str, slot: str, last_error: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE provider_keys SET last_error=?, updated_at=? WHERE provider_id=? AND key_slot=?
                """,
                (str(last_error), _utc_now_iso(), str(provider_id), str(slot)),
            )

    def get_state(self, *, provider_id: str) -> Dict[str, Any]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM provider_state WHERE provider_id=?", (str(provider_id),)).fetchone()
        if row is None:
            return {
                "provider_id": str(provider_id),
                "health_state": ProviderHealthState.HEALTHY,
                "consecutive_quota_errors": 0,
                "window_started_at": None,
                "fallback_level": "none",
                "fallback_reason": None,
                "active_slot": "primary",
                "last_error": None,
                "last_transition_at": _utc_now_iso(),
                "next_recovery_check_at": None,
                "quota_signal": None,
            }
        return {
            "provider_id": str(row["provider_id"]),
            "health_state": str(row["health_state"]),
            "consecutive_quota_errors": int(row["consecutive_quota_errors"] or 0),
            "window_started_at": (str(row["window_started_at"]) if row["window_started_at"] else None),
            "fallback_level": str(row["fallback_level"]),
            "fallback_reason": (str(row["fallback_reason"]) if row["fallback_reason"] else None),
            "active_slot": (str(row["active_slot"]) if row["active_slot"] else None),
            "last_error": (str(row["last_error"]) if row["last_error"] else None),
            "last_transition_at": str(row["last_transition_at"]),
            "next_recovery_check_at": (str(row["next_recovery_check_at"]) if row["next_recovery_check_at"] else None),
            "quota_signal": (str(row["quota_signal"]) if row["quota_signal"] else None),
        }

    def upsert_state(self, *, provider_id: str, patch: Mapping[str, Any]) -> Dict[str, Any]:
        current = self.get_state(provider_id=provider_id)
        merged = dict(current)
        for key, value in dict(patch or {}).items():
            merged[str(key)] = value
        merged["provider_id"] = str(provider_id)
        merged["last_transition_at"] = _utc_now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO provider_state(
                  provider_id, health_state, consecutive_quota_errors, window_started_at, fallback_level, fallback_reason,
                  active_slot, last_error, last_transition_at, next_recovery_check_at, quota_signal
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider_id) DO UPDATE SET
                  health_state=excluded.health_state,
                  consecutive_quota_errors=excluded.consecutive_quota_errors,
                  window_started_at=excluded.window_started_at,
                  fallback_level=excluded.fallback_level,
                  fallback_reason=excluded.fallback_reason,
                  active_slot=excluded.active_slot,
                  last_error=excluded.last_error,
                  last_transition_at=excluded.last_transition_at,
                  next_recovery_check_at=excluded.next_recovery_check_at,
                  quota_signal=excluded.quota_signal
                """,
                (
                    merged["provider_id"],
                    str(merged.get("health_state", ProviderHealthState.HEALTHY)),
                    int(merged.get("consecutive_quota_errors", 0) or 0),
                    merged.get("window_started_at"),
                    str(merged.get("fallback_level", "none")),
                    merged.get("fallback_reason"),
                    merged.get("active_slot"),
                    merged.get("last_error"),
                    str(merged.get("last_transition_at", _utc_now_iso())),
                    merged.get("next_recovery_check_at"),
                    merged.get("quota_signal"),
                ),
            )
        return self.get_state(provider_id=provider_id)

    def write_setting(self, *, key: str, value: Mapping[str, Any] | str) -> None:
        if isinstance(value, Mapping):
            raw = json.dumps(dict(value), ensure_ascii=False, sort_keys=True)
        else:
            raw = str(value)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO orchestrator_settings(setting_key, setting_value, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(setting_key) DO UPDATE SET setting_value=excluded.setting_value, updated_at=excluded.updated_at
                """,
                (str(key), raw, _utc_now_iso()),
            )

    def read_setting(self, *, key: str) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT setting_value FROM orchestrator_settings WHERE setting_key=?",
                (str(key),),
            ).fetchone()
        if row is None:
            return None
        return str(row["setting_value"])

    def record_audit(self, *, actor: str, event_type: str, state_diff: Mapping[str, Any], reason: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_events(ts, actor, event_type, state_diff_json, reason) VALUES(?, ?, ?, ?, ?)
                """,
                (_utc_now_iso(), str(actor), str(event_type), json.dumps(dict(state_diff), ensure_ascii=False), str(reason)),
            )

    def list_recent_audit(self, *, limit: int = 100) -> List[Dict[str, Any]]:
        lim = max(1, int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, actor, event_type, state_diff_json, reason FROM audit_events ORDER BY id DESC LIMIT ?",
                (lim,),
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            diff = {}
            try:
                diff = dict(json.loads(str(row["state_diff_json"]) or "{}"))
            except Exception:  # noqa: BLE001
                diff = {}
            out.append(
                {
                    "ts": str(row["ts"]),
                    "actor": str(row["actor"]),
                    "event_type": str(row["event_type"]),
                    "state_diff": diff,
                    "reason": str(row["reason"] or ""),
                }
            )
        return out


def _extract_http_code_from_error(err: str) -> Optional[int]:
    text = str(err or "")
    marker = "HTTP "
    idx = text.find(marker)
    if idx < 0:
        return None
    rest = text[idx + len(marker) : idx + len(marker) + 3]
    try:
        return int(rest)
    except Exception:  # noqa: BLE001
        return None


def _classify_health_signal(*, error: Optional[str], quota_signal: Optional[str]) -> Tuple[str, str]:
    if error is None:
        if quota_signal == "low_remaining":
            return ProviderHealthState.WARNING, "quota_warning"
        return ProviderHealthState.HEALTHY, "ok"
    code = _extract_http_code_from_error(error)
    if code in {402, 403}:
        return ProviderHealthState.EXHAUSTED, "quota_exhausted"
    if code == 429:
        return ProviderHealthState.WARNING, "rate_limited"
    if code is not None and code >= 500:
        return ProviderHealthState.WARNING, "provider_outage"
    if "missing_env" in str(error).lower():
        return ProviderHealthState.EXHAUSTED, "key_invalid"
    return ProviderHealthState.WARNING, "provider_error"


class OrchestratorRuntime:
    def __init__(self, *, config: OrchestratorConfig, actor: str = "system") -> None:
        self.config = config
        self.actor = str(actor)
        self.vault = ProviderKeyVault(sqlite_path=config.sqlite_path, master_key_env=config.master_key_env)
        self._alert_lock = threading.Lock()
        self._last_alert: Dict[Tuple[str, str], float] = {}
        self._silence_until_epoch = self._read_silence_until_epoch()
        self._last_emitted_alert: Optional[OrchestratorAlertEvent] = None

    def _read_silence_until_epoch(self) -> float:
        raw = self.vault.read_setting(key="alerts.silence_until")
        if not raw:
            return 0.0
        try:
            dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
            return dt.timestamp()
        except Exception:  # noqa: BLE001
            return 0.0

    def set_silence(self, *, duration_sec: int) -> Dict[str, Any]:
        until = _utc_now() + timedelta(seconds=max(1, int(duration_sec)))
        until_iso = until.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.vault.write_setting(key="alerts.silence_until", value=until_iso)
        self._silence_until_epoch = until.timestamp()
        self.vault.record_audit(actor=self.actor, event_type="alerts_silence", state_diff={"until": until_iso}, reason="manual")
        return {"status": "ok", "silenced_until": until_iso}

    def configure_webhook(self, *, webhook_url: str, types: Sequence[str]) -> Dict[str, Any]:
        payload = {
            "url": str(webhook_url).strip(),
            "types": sorted({str(x).strip() for x in list(types) if str(x).strip()}),
            "updated_at": _utc_now_iso(),
        }
        self.vault.write_setting(key="alerts.webhook", value=payload)
        self.vault.record_audit(actor=self.actor, event_type="alerts_configure", state_diff=payload, reason="manual")
        return {"status": "ok", "webhook": {"url": payload["url"], "types": list(payload["types"])}}

    def test_webhook(self) -> Dict[str, Any]:
        cfg_raw = self.vault.read_setting(key="alerts.webhook")
        if not cfg_raw:
            raise ValueError("webhook_not_configured")
        cfg = dict(json.loads(cfg_raw))
        url = str(cfg.get("url", "")).strip()
        if not url:
            raise ValueError("webhook_not_configured")
        payload = {
            "event_type": "alerts_test",
            "status": "ok",
            "provider": "n/a",
            "message": "omega orchestrator alerts test",
            "timestamp": _utc_now_iso(),
        }
        req = urlrequest.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=8.0) as resp:  # noqa: S310
                _ = resp.read()
        except urlerror.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"http_error:{exc.code}:{body}") from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f"url_error:{exc}") from exc
        return {"status": "ok", "channel": "webhook"}

    def _emit_alert(self, event: OrchestratorAlertEvent) -> None:
        now_epoch = _utc_now().timestamp()
        if now_epoch < float(self._silence_until_epoch):
            return
        key = (str(event.provider_id), str(event.alert_type))
        with self._alert_lock:
            prev = self._last_alert.get(key, 0.0)
            if (now_epoch - prev) < float(self.config.alerts_cooldown_sec):
                return
            self._last_alert[key] = now_epoch
        self._last_emitted_alert = event
        self.vault.record_audit(
            actor=self.actor,
            event_type=f"alert:{event.alert_type}",
            state_diff={
                "provider_id": event.provider_id,
                "protection_mode": event.protection_mode,
                "reason": event.reason,
                "message": event.message,
                "details": dict(event.details),
            },
            reason=event.reason,
        )
        cfg_raw = self.vault.read_setting(key="alerts.webhook")
        if not cfg_raw:
            return
        try:
            cfg = dict(json.loads(cfg_raw))
        except Exception:  # noqa: BLE001
            return
        allowed = {str(x).strip() for x in list(cfg.get("types", [])) if str(x).strip()}
        if allowed and str(event.alert_type) not in allowed:
            return
        url = str(cfg.get("url", "")).strip()
        if not url:
            return
        payload = {
            "event_type": str(event.alert_type),
            "provider_id": str(event.provider_id),
            "status": str(event.reason),
            "protection_mode": str(event.protection_mode),
            "message": str(event.message),
            "timestamp": str(event.ts),
            "details": dict(event.details),
        }
        attempts = 0
        while attempts < 3:
            attempts += 1
            req = urlrequest.Request(
                url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlrequest.urlopen(req, timeout=8.0) as resp:  # noqa: S310
                    _ = resp.read()
                return
            except Exception as exc:  # noqa: BLE001
                if attempts >= 3:
                    LOGGER.warning("orchestrator webhook delivery failed: %s", exc)
                    return

    def set_fallback_mode(self, *, mode: str) -> Dict[str, Any]:
        m = str(mode).strip().lower()
        if m not in {"rule_only", "fail_closed"}:
            raise ValueError("fallback_mode must be rule_only|fail_closed")
        self.vault.write_setting(key="fallback.mode", value=m)
        self.vault.record_audit(actor=self.actor, event_type="fallback_mode_set", state_diff={"mode": m}, reason="manual")
        return {"status": "ok", "mode": m}

    def set_fallback_threshold(self, *, errors: int, window_sec: int) -> Dict[str, Any]:
        payload = {"errors": max(1, int(errors)), "window_sec": max(1, int(window_sec))}
        self.vault.write_setting(key="fallback.threshold", value=payload)
        self.vault.record_audit(actor=self.actor, event_type="fallback_threshold_set", state_diff=payload, reason="manual")
        return {"status": "ok", "threshold": payload}

    def effective_fallback_mode(self) -> str:
        stored = self.vault.read_setting(key="fallback.mode")
        if stored:
            mode = str(stored).strip().lower()
            if mode in {"rule_only", "fail_closed"}:
                return mode
        return str(self.config.fallback_mode)

    def effective_threshold(self) -> Tuple[int, int]:
        stored = self.vault.read_setting(key="fallback.threshold")
        if stored:
            try:
                raw = dict(json.loads(stored))
                return max(1, int(raw.get("errors", self.config.threshold_errors))), max(
                    1, int(raw.get("window_sec", self.config.threshold_window_sec))
                )
            except Exception:  # noqa: BLE001
                pass
        return int(self.config.threshold_errors), int(self.config.threshold_window_sec)

    def _provider_candidates(self) -> List[ProviderCandidate]:
        return list(self.config.providers)

    def resolve_route(self) -> List[ProviderCandidate]:
        out: List[ProviderCandidate] = []
        for row in self._provider_candidates():
            state = self.vault.get_state(provider_id=row.provider_id)
            health = str(state.get("health_state", ProviderHealthState.HEALTHY))
            active_slot = str(state.get("active_slot", "primary") or "primary")
            if health == ProviderHealthState.EXHAUSTED and row.key_slot == "primary":
                continue
            if health == ProviderHealthState.FALLBACK_ACTIVE and row.key_slot == "primary" and active_slot != "primary":
                continue
            out.append(row)
        if not out:
            out = list(self._provider_candidates())
        return out

    def should_probe_recovery(self, *, provider_id: str) -> bool:
        state = self.vault.get_state(provider_id=provider_id)
        next_check = str(state.get("next_recovery_check_at") or "").strip()
        if not next_check:
            return True
        try:
            return _utc_now() >= datetime.fromisoformat(next_check.replace("Z", "+00:00"))
        except Exception:  # noqa: BLE001
            return True

    def mark_success(self, *, provider_id: str, slot: str) -> None:
        prev = self.vault.get_state(provider_id=provider_id)
        now = _utc_now()
        next_check = (now + timedelta(seconds=max(120, int(self.config.healthcheck_interval_sec)))).strftime("%Y-%m-%dT%H:%M:%SZ")
        patch = {
            "health_state": ProviderHealthState.HEALTHY,
            "consecutive_quota_errors": 0,
            "window_started_at": None,
            "fallback_level": "none" if slot == "primary" else "backup_provider",
            "fallback_reason": None,
            "active_slot": str(slot),
            "last_error": None,
            "next_recovery_check_at": next_check,
            "quota_signal": None,
        }
        self.vault.upsert_state(provider_id=provider_id, patch=patch)
        if str(prev.get("health_state")) in {ProviderHealthState.FALLBACK_ACTIVE, ProviderHealthState.RECOVERING, ProviderHealthState.EXHAUSTED}:
            self._emit_alert(
                OrchestratorAlertEvent(
                    alert_type="fallback_recovered",
                    provider_id=provider_id,
                    message="LLM provider recovered and primary path is healthy.",
                    protection_mode="llm",
                    reason="recovered",
                    ts=_utc_now_iso(),
                    details={"slot": str(slot)},
                )
            )

    def mark_warning(self, *, provider_id: str, slot: str, reason: str) -> None:
        state = self.vault.upsert_state(
            provider_id=provider_id,
            patch={
                "health_state": ProviderHealthState.WARNING,
                "active_slot": str(slot),
                "fallback_level": "none",
                "fallback_reason": None,
                "last_error": None,
                "quota_signal": str(reason),
            },
        )
        _ = state
        self._emit_alert(
            OrchestratorAlertEvent(
                alert_type="quota_warning",
                provider_id=provider_id,
                message=f"Provider {provider_id} reported low remaining quota.",
                protection_mode="llm",
                reason=str(reason),
                ts=_utc_now_iso(),
                details={"slot": str(slot)},
            )
        )

    def mark_error(self, *, provider_id: str, slot: str, error: str, quota_signal: Optional[str] = None) -> Dict[str, Any]:
        errors_threshold, window_threshold = self.effective_threshold()
        state = self.vault.get_state(provider_id=provider_id)
        now = _utc_now()
        window_started_at = state.get("window_started_at")
        count = int(state.get("consecutive_quota_errors", 0) or 0)
        if window_started_at:
            try:
                started = datetime.fromisoformat(str(window_started_at).replace("Z", "+00:00"))
            except Exception:  # noqa: BLE001
                started = now
            if (now - started).total_seconds() > float(window_threshold):
                count = 0
                window_started_at = None
        if window_started_at is None:
            window_started_at = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        health_signal, reason = _classify_health_signal(error=error, quota_signal=quota_signal)
        if reason in {"quota_exhausted", "rate_limited"}:
            count += 1
        else:
            count = max(1, count + 1)
        health_state = health_signal
        fallback_level = "none"
        fallback_reason = None
        if count >= int(errors_threshold) and reason in {"quota_exhausted", "rate_limited", "provider_outage", "provider_error", "key_invalid"}:
            health_state = ProviderHealthState.FALLBACK_ACTIVE
            fallback_level = "backup_provider" if slot == "primary" else "rule_only"
            fallback_reason = reason
        elif health_signal == ProviderHealthState.WARNING:
            fallback_level = "none"
        next_check = (now + timedelta(seconds=max(120, int(self.config.healthcheck_interval_sec)))).strftime("%Y-%m-%dT%H:%M:%SZ")
        patch = {
            "health_state": health_state,
            "consecutive_quota_errors": int(count),
            "window_started_at": str(window_started_at),
            "fallback_level": fallback_level,
            "fallback_reason": fallback_reason,
            "active_slot": str(slot),
            "last_error": str(error),
            "next_recovery_check_at": next_check,
            "quota_signal": (str(quota_signal) if quota_signal else None),
        }
        updated = self.vault.upsert_state(provider_id=provider_id, patch=patch)
        self.vault.set_key_error(provider_id=provider_id, slot=slot, last_error=str(error))
        alert_type = None
        if reason == "quota_exhausted":
            alert_type = "quota_exhausted"
        elif reason == "key_invalid":
            alert_type = "key_invalid"
        elif reason == "provider_outage":
            alert_type = "provider_outage"
        elif reason == "rate_limited":
            alert_type = "quota_warning"
        if alert_type:
            self._emit_alert(
                OrchestratorAlertEvent(
                    alert_type=alert_type,
                    provider_id=provider_id,
                    message=f"Provider {provider_id} degraded: {reason}",
                    protection_mode=("rule_only" if health_state == ProviderHealthState.FALLBACK_ACTIVE and slot == "backup" else "llm"),
                    reason=reason,
                    ts=_utc_now_iso(),
                    details={"slot": slot, "error": str(error), "consecutive_errors": int(count)},
                )
            )
        if health_state == ProviderHealthState.FALLBACK_ACTIVE:
            self._emit_alert(
                OrchestratorAlertEvent(
                    alert_type="fallback_activated",
                    provider_id=provider_id,
                    message=f"Fallback activated for provider {provider_id}: {reason}",
                    protection_mode=("rule_only" if slot == "backup" else "backup_provider"),
                    reason=reason,
                    ts=_utc_now_iso(),
                    details={"slot": slot, "fallback_level": str(fallback_level)},
                )
            )
        return updated

    def get_key_for_candidate(self, *, candidate: ProviderCandidate) -> Optional[str]:
        return self.vault.get_key(provider_id=candidate.provider_id, slot=candidate.key_slot)

    def status_snapshot(self) -> Dict[str, Any]:
        providers: Dict[str, Any] = {}
        for cand in self._provider_candidates():
            providers.setdefault(
                cand.provider_id,
                {
                    "provider_id": cand.provider_id,
                    "provider_type": cand.provider_type,
                    "model": cand.model,
                    "base_url": cand.base_url,
                    "state": self.vault.get_state(provider_id=cand.provider_id),
                },
            )
        return {
            "enabled": bool(self.config.enabled),
            "fallback_mode": self.effective_fallback_mode(),
            "threshold": {"errors": self.effective_threshold()[0], "window_sec": self.effective_threshold()[1]},
            "recovery_interval_sec": int(self.config.healthcheck_interval_sec),
            "alerts_cooldown_sec": int(self.config.alerts_cooldown_sec),
            "providers": list(providers.values()),
            "keys": self.vault.list_keys(),
            "last_alert": (self._last_emitted_alert.__dict__ if self._last_emitted_alert is not None else None),
        }

    def add_primary_key(self, *, provider_id: str, key: str) -> Dict[str, Any]:
        self.vault.put_key(provider_id=provider_id, slot="primary", key_ref=provider_id, key_plain=key, status="active")
        self.vault.record_audit(actor=self.actor, event_type="key_add", state_diff={"provider_id": provider_id, "slot": "primary"}, reason="manual")
        return {"status": "ok", "provider_id": str(provider_id), "slot": "primary", "key_mask": mask_secret(key)}

    def set_backup_key(self, *, provider_id: str, key: str) -> Dict[str, Any]:
        self.vault.put_key(provider_id=provider_id, slot="backup", key_ref=provider_id, key_plain=key, status="active")
        self.vault.record_audit(actor=self.actor, event_type="key_add", state_diff={"provider_id": provider_id, "slot": "backup"}, reason="manual")
        return {"status": "ok", "provider_id": str(provider_id), "slot": "backup", "key_mask": mask_secret(key)}

    def rotate_key(self, *, provider_id: str, key: str) -> Dict[str, Any]:
        self.vault.rotate_primary(provider_id=provider_id, key_ref=provider_id, key_plain=key)
        self.vault.record_audit(actor=self.actor, event_type="key_rotate", state_diff={"provider_id": provider_id}, reason="manual")
        return {"status": "ok", "provider_id": str(provider_id), "slot": "primary", "key_mask": mask_secret(key)}

    def validate_key(self, *, provider_id: str, slot: str = "primary") -> Dict[str, Any]:
        secret = self.vault.get_key(provider_id=provider_id, slot=slot)
        if not secret:
            return {"status": "error", "provider_id": provider_id, "slot": slot, "reason": "key_not_found"}
        candidate = None
        for row in self._provider_candidates():
            if row.provider_id == provider_id:
                candidate = row
                break
        if candidate is None:
            return {"status": "error", "provider_id": provider_id, "slot": slot, "reason": "provider_not_configured"}
        ok, reason = self._probe_provider(candidate=candidate, api_key=secret)
        if not ok:
            self.vault.set_key_error(provider_id=provider_id, slot=slot, last_error=reason)
            self._emit_alert(
                OrchestratorAlertEvent(
                    alert_type="key_invalid",
                    provider_id=provider_id,
                    message=f"Key validation failed for provider={provider_id} slot={slot}",
                    protection_mode="llm",
                    reason=reason,
                    ts=_utc_now_iso(),
                    details={"slot": slot},
                )
            )
            return {"status": "error", "provider_id": provider_id, "slot": slot, "reason": reason}
        self.vault.set_key_error(provider_id=provider_id, slot=slot, last_error="")
        return {"status": "ok", "provider_id": provider_id, "slot": slot}

    @staticmethod
    def _probe_provider(*, candidate: ProviderCandidate, api_key: str) -> Tuple[bool, str]:
        provider_type = str(candidate.provider_type).strip().lower()
        base = str(candidate.base_url).rstrip("/")
        model = str(candidate.model)
        try:
            if provider_type == "anthropic":
                payload = {
                    "model": model,
                    "max_tokens": 8,
                    "temperature": 0,
                    "messages": [{"role": "user", "content": "healthcheck"}],
                }
                req = urlrequest.Request(
                    f"{base}/messages",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "x-api-key": str(api_key),
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    method="POST",
                )
            else:
                payload = {
                    "model": model,
                    "response_format": {"type": "json_object"},
                    "messages": [{"role": "user", "content": '{"health":"ok"}'}],
                    "temperature": 0,
                }
                req = urlrequest.Request(
                    f"{base}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
            with urlrequest.urlopen(req, timeout=10.0) as resp:  # noqa: S310
                code = int(getattr(resp, "status", 200))
                _ = resp.read()
            if code >= 400:
                return False, f"http_status:{code}"
            return True, "ok"
        except urlerror.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                body = str(exc)
            return False, f"HTTP {int(exc.code)}: {body}"
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
