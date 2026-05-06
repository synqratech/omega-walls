from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
import secrets
import sqlite3
import threading
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from omega.api.incident_export import IncidentExportStore
from omega.telemetry.redaction import redact_text


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)


def _uuid7_str() -> str:
    import uuid

    candidate = getattr(uuid, "uuid7", None)
    if callable(candidate):
        return str(candidate())
    return str(uuid.uuid4())


def _short_id(prefix: str) -> str:
    return f"{prefix}_{_uuid7_str().replace('-', '')[:16]}"


def _sanitize_redacted(text: str, *, max_chars: int = 1200) -> str:
    red = redact_text(str(text or ""), max_chars=max_chars)
    return str(red.redacted)


def _infer_model_version(runtime_refs: Mapping[str, Any]) -> str:
    provider = str(runtime_refs.get("provider", "")).strip()
    model = str(runtime_refs.get("model", "")).strip()
    if provider and model:
        return f"{provider}:{model}"
    return model or provider or "unknown"


def _derive_trust_level(source_type: str) -> str:
    low = str(source_type or "").strip().lower()
    if low in {"web", "email", "pdf", "api_response", "tool_output"}:
        return "untrusted"
    if low in {"memory_carryover"}:
        return "verified"
    return "system"


def _retention_cutoff_iso(days: int) -> str:
    dt = _utc_now() - timedelta(days=max(1, int(days)))
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


class ReplayGenerateRequest(BaseModel):
    redact_sensitive: bool = True
    include_raw_context: bool = False
    max_steps: int = Field(default=50, ge=1, le=50)
    format: str = Field(default="json_manifest", pattern="^json_manifest$")


class ReplayGenerateResponse(BaseModel):
    job_id: str
    status: str = Field(pattern="^(pending|processing|completed|failed)$")


class ReplayJobResponse(BaseModel):
    job_id: str
    incident_id: str
    status: str = Field(pattern="^(pending|processing|completed|failed)$")
    created_at: str
    updated_at: str
    download_url: Optional[str] = None
    expires_at: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class IncidentReplayConfig:
    enabled: bool
    contract_version: str
    store_path: str
    package_storage_path: str
    encryption_key_env: str
    download_ttl_hours: int
    job_ttl_hours: int
    max_steps: int
    worker_max_concurrent_jobs: int
    required_scope_read: str
    required_scope_raw: str
    redact_by_default: bool

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "IncidentReplayConfig":
        data = dict(cfg or {})
        store_cfg = data.get("store", {}) if isinstance(data.get("store", {}), Mapping) else {}
        storage_cfg = data.get("package_storage", {}) if isinstance(data.get("package_storage", {}), Mapping) else {}
        worker_cfg = data.get("worker", {}) if isinstance(data.get("worker", {}), Mapping) else {}
        auth_cfg = data.get("auth", {}) if isinstance(data.get("auth", {}), Mapping) else {}
        scopes = auth_cfg.get("required_scopes", {}) if isinstance(auth_cfg.get("required_scopes", {}), Mapping) else {}
        return cls(
            enabled=bool(data.get("enabled", False)),
            contract_version=str(data.get("contract_version", "1.0.0")).strip() or "1.0.0",
            store_path=str(store_cfg.get("sqlite_path", "artifacts/state/incident_replay.db")).strip()
            or "artifacts/state/incident_replay.db",
            package_storage_path=str(storage_cfg.get("path", "artifacts/replay/packages")).strip()
            or "artifacts/replay/packages",
            encryption_key_env=str(storage_cfg.get("encryption_key_env", "OMEGA_REPLAY_ENCRYPTION_KEY")).strip()
            or "OMEGA_REPLAY_ENCRYPTION_KEY",
            download_ttl_hours=max(1, int(data.get("download_ttl_hours", 24))),
            job_ttl_hours=max(1, int(data.get("job_ttl_hours", 72))),
            max_steps=max(1, min(50, int(data.get("max_steps", 50)))),
            worker_max_concurrent_jobs=max(1, int(worker_cfg.get("max_concurrent_jobs", 4))),
            required_scope_read=str(scopes.get("read", "incidents:replay:read")).strip() or "incidents:replay:read",
            required_scope_raw=str(scopes.get("raw", "incidents:replay:raw")).strip() or "incidents:replay:raw",
            redact_by_default=bool(data.get("redact_by_default", True)),
        )


class IncidentReplayStore:
    def __init__(self, *, sqlite_path: str | Path) -> None:
        self.sqlite_path = Path(str(sqlite_path))
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
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
                CREATE TABLE IF NOT EXISTS replay_jobs (
                  job_id TEXT PRIMARY KEY,
                  incident_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  include_raw_context INTEGER NOT NULL,
                  redact_sensitive INTEGER NOT NULL,
                  max_steps INTEGER NOT NULL,
                  format TEXT NOT NULL,
                  requested_by_key_id TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  started_at TEXT,
                  completed_at TEXT,
                  error TEXT,
                  expires_at TEXT NOT NULL,
                  download_token TEXT,
                  download_expires_at TEXT,
                  package_path TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_replay_jobs_incident_id ON replay_jobs(incident_id);
                CREATE INDEX IF NOT EXISTS idx_replay_jobs_status ON replay_jobs(status);
                CREATE INDEX IF NOT EXISTS idx_replay_jobs_expires_at ON replay_jobs(expires_at);

                CREATE TABLE IF NOT EXISTS replay_download_tokens (
                  token TEXT PRIMARY KEY,
                  job_id TEXT NOT NULL,
                  key_id TEXT NOT NULL,
                  package_path TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  used_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_replay_download_tokens_expires_at ON replay_download_tokens(expires_at);
                CREATE INDEX IF NOT EXISTS idx_replay_download_tokens_job_id ON replay_download_tokens(job_id);

                CREATE TABLE IF NOT EXISTS replay_audit_log (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT NOT NULL,
                  key_hash TEXT NOT NULL,
                  endpoint TEXT NOT NULL,
                  status_code INTEGER NOT NULL,
                  ip TEXT NOT NULL,
                  incident_id TEXT,
                  job_id TEXT,
                  scope TEXT NOT NULL,
                  action TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_replay_audit_log_ts ON replay_audit_log(ts);
                CREATE INDEX IF NOT EXISTS idx_replay_audit_log_incident_id ON replay_audit_log(incident_id);
                CREATE INDEX IF NOT EXISTS idx_replay_audit_log_job_id ON replay_audit_log(job_id);
                """
            )

    def create_job(
        self,
        *,
        incident_id: str,
        key_id: str,
        include_raw_context: bool,
        redact_sensitive: bool,
        max_steps: int,
        out_format: str,
        job_ttl_hours: int,
    ) -> Dict[str, Any]:
        now = _utc_now_iso()
        expires_at = (_utc_now() + timedelta(hours=max(1, int(job_ttl_hours)))).strftime("%Y-%m-%dT%H:%M:%SZ")
        job_id = _uuid7_str()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO replay_jobs(
                  job_id, incident_id, status, include_raw_context, redact_sensitive, max_steps, format,
                  requested_by_key_id, created_at, updated_at, expires_at
                ) VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    str(incident_id),
                    1 if bool(include_raw_context) else 0,
                    1 if bool(redact_sensitive) else 0,
                    int(max_steps),
                    str(out_format),
                    str(key_id),
                    now,
                    now,
                    expires_at,
                ),
            )
        return self.get_job(job_id=job_id) or {}

    def update_job_status(self, *, job_id: str, status: str, error: Optional[str] = None) -> None:
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            if status == "processing":
                conn.execute(
                    "UPDATE replay_jobs SET status=?, updated_at=?, started_at=?, error=NULL WHERE job_id=?",
                    (str(status), now, now, str(job_id)),
                )
            elif status == "failed":
                conn.execute(
                    "UPDATE replay_jobs SET status=?, updated_at=?, completed_at=?, error=? WHERE job_id=?",
                    (str(status), now, now, str(error or "replay_job_failed"), str(job_id)),
                )
            else:
                conn.execute(
                    "UPDATE replay_jobs SET status=?, updated_at=? WHERE job_id=?",
                    (str(status), now, str(job_id)),
                )

    def mark_job_completed(
        self,
        *,
        job_id: str,
        key_id: str,
        package_path: str,
        download_ttl_hours: int,
    ) -> Tuple[str, str]:
        now = _utc_now()
        now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
        expires_at = (now + timedelta(hours=max(1, int(download_ttl_hours)))).strftime("%Y-%m-%dT%H:%M:%SZ")
        token = secrets.token_urlsafe(36)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO replay_download_tokens(token, job_id, key_id, package_path, expires_at, created_at, used_at)
                VALUES (?, ?, ?, ?, ?, ?, NULL)
                """,
                (token, str(job_id), str(key_id), str(package_path), expires_at, now_iso),
            )
            conn.execute(
                """
                UPDATE replay_jobs
                SET status='completed', updated_at=?, completed_at=?, error=NULL,
                    download_token=?, download_expires_at=?, package_path=?
                WHERE job_id=?
                """,
                (now_iso, now_iso, token, expires_at, str(package_path), str(job_id)),
            )
        return token, expires_at

    def get_job(self, *, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM replay_jobs WHERE job_id=?", (str(job_id),)).fetchone()
        if row is None:
            return None
        return {
            "job_id": str(row["job_id"]),
            "incident_id": str(row["incident_id"]),
            "status": str(row["status"]),
            "include_raw_context": bool(int(row["include_raw_context"])),
            "redact_sensitive": bool(int(row["redact_sensitive"])),
            "max_steps": int(row["max_steps"]),
            "format": str(row["format"]),
            "requested_by_key_id": str(row["requested_by_key_id"]),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "started_at": (str(row["started_at"]) if row["started_at"] is not None else None),
            "completed_at": (str(row["completed_at"]) if row["completed_at"] is not None else None),
            "error": (str(row["error"]) if row["error"] is not None else None),
            "expires_at": str(row["expires_at"]),
            "download_token": (str(row["download_token"]) if row["download_token"] is not None else None),
            "download_expires_at": (
                str(row["download_expires_at"]) if row["download_expires_at"] is not None else None
            ),
            "package_path": (str(row["package_path"]) if row["package_path"] is not None else None),
        }

    def consume_download_token(self, *, token: str, key_id: str) -> Optional[Dict[str, Any]]:
        now_iso = _utc_now_iso()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM replay_download_tokens WHERE token=?", (str(token),)).fetchone()
            if row is None:
                return None
            if str(row["key_id"]) != str(key_id):
                return None
            if row["used_at"] is not None:
                return None
            if str(row["expires_at"]) < now_iso:
                return None
            conn.execute("UPDATE replay_download_tokens SET used_at=? WHERE token=?", (now_iso, str(token)))
        return {
            "token": str(row["token"]),
            "job_id": str(row["job_id"]),
            "package_path": str(row["package_path"]),
            "expires_at": str(row["expires_at"]),
        }

    def cleanup_expired(self) -> List[str]:
        now_iso = _utc_now_iso()
        remove_paths: List[str] = []
        with self._lock, self._connect() as conn:
            expired_tokens = conn.execute(
                "SELECT package_path FROM replay_download_tokens WHERE expires_at < ? OR used_at IS NOT NULL",
                (now_iso,),
            ).fetchall()
            remove_paths.extend(str(row["package_path"]) for row in expired_tokens if str(row["package_path"]).strip())
            conn.execute("DELETE FROM replay_download_tokens WHERE expires_at < ? OR used_at IS NOT NULL", (now_iso,))

            expired_jobs = conn.execute(
                "SELECT package_path FROM replay_jobs WHERE expires_at < ?",
                (now_iso,),
            ).fetchall()
            remove_paths.extend(str(row["package_path"]) for row in expired_jobs if row["package_path"] is not None)
            conn.execute("DELETE FROM replay_jobs WHERE expires_at < ?", (now_iso,))
        return sorted(set(remove_paths))

    def log_access(
        self,
        *,
        key_hash: str,
        endpoint: str,
        status_code: int,
        ip: str,
        incident_id: Optional[str],
        job_id: Optional[str],
        scope: str,
        action: str,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO replay_audit_log(ts, key_hash, endpoint, status_code, ip, incident_id, job_id, scope, action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _utc_now_iso(),
                    str(key_hash),
                    str(endpoint),
                    int(status_code),
                    str(ip),
                    (str(incident_id) if incident_id is not None else None),
                    (str(job_id) if job_id is not None else None),
                    str(scope),
                    str(action),
                ),
            )


class _ReplayCrypto:
    def __init__(self, *, key_material: str) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        km = str(key_material or "").strip()
        if not km:
            raise ValueError("missing_replay_encryption_key")
        key_bytes: bytes
        if km.startswith("base64:"):
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


class IncidentReplayJobManager:
    def __init__(
        self,
        *,
        config: IncidentReplayConfig,
        replay_store: IncidentReplayStore,
        incident_store: IncidentExportStore,
        incident_retention_days: int,
    ) -> None:
        self.config = config
        self.replay_store = replay_store
        self.incident_store = incident_store
        self.incident_retention_days = max(1, int(incident_retention_days))
        self.package_root = Path(str(config.package_storage_path))
        self.package_root.mkdir(parents=True, exist_ok=True)
        key_material = str(os.environ.get(str(config.encryption_key_env), "")).strip()
        if not key_material:
            raise ValueError(f"missing_replay_encryption_key_env:{config.encryption_key_env}")
        self.crypto = _ReplayCrypto(key_material=key_material)
        self._semaphore = asyncio.Semaphore(max(1, int(config.worker_max_concurrent_jobs)))
        self._tasks: Dict[str, asyncio.Task[Any]] = {}
        self._guard = threading.Lock()

    def submit(self, *, job_id: str) -> None:
        self._cleanup_artifacts()
        task = asyncio.create_task(self._process_job(job_id=job_id))
        with self._guard:
            self._tasks[job_id] = task

    def _cleanup_artifacts(self) -> None:
        for raw in self.replay_store.cleanup_expired():
            try:
                path = Path(str(raw))
                if not path.is_absolute():
                    path = (self.package_root / path).resolve()
                if path.exists() and self.package_root in path.parents:
                    path.unlink(missing_ok=True)
            except Exception:
                continue

    async def _process_job(self, *, job_id: str) -> None:
        async with self._semaphore:
            try:
                self.replay_store.update_job_status(job_id=job_id, status="processing")
                job = self.replay_store.get_job(job_id=job_id)
                if job is None:
                    return
                incident_id = str(job["incident_id"])
                incident = self.incident_store.get_incident(incident_id=incident_id)
                if incident is None:
                    raise ValueError("incident_not_found_or_expired")
                manifest = self._build_manifest(incident=incident, job=job)
                payload = json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                encrypted = self.crypto.encrypt(payload)
                package_file = self.package_root / f"{_short_id('replay')}.enc"
                package_file.parent.mkdir(parents=True, exist_ok=True)
                package_file.write_bytes(encrypted)
                self.replay_store.mark_job_completed(
                    job_id=job_id,
                    key_id=str(job["requested_by_key_id"]),
                    package_path=str(package_file),
                    download_ttl_hours=int(self.config.download_ttl_hours),
                )
            except Exception as exc:  # noqa: BLE001
                self.replay_store.update_job_status(job_id=job_id, status="failed", error=str(exc))
            finally:
                with self._guard:
                    self._tasks.pop(job_id, None)

    def _build_manifest(self, *, incident: Mapping[str, Any], job: Mapping[str, Any]) -> Dict[str, Any]:
        include_raw = bool(job.get("include_raw_context", False))
        redact_sensitive = bool(job.get("redact_sensitive", True))
        max_steps = max(1, min(int(job.get("max_steps", self.config.max_steps)), int(self.config.max_steps)))
        chain = list(incident.get("chain_of_events", []) or [])
        if len(chain) > max_steps:
            chain = chain[-max_steps:]
        policy_triggered = str(incident.get("policy_triggered", "none"))
        threshold = dict(incident.get("threshold_config", {}) or {})
        resolution = dict(incident.get("resolution", {}) or {})
        steps: List[Dict[str, Any]] = []
        risk_acc = 0.0
        for idx, row in enumerate(chain, start=1):
            source_hash = str((row or {}).get("source_hash", ""))
            snippet = str((row or {}).get("snippet_redacted", ""))
            payload = str((row or {}).get("tool_call_payload_redacted", ""))
            if redact_sensitive:
                snippet = _sanitize_redacted(snippet, max_chars=1800)
                payload = _sanitize_redacted(payload, max_chars=1200)
            risk_delta = float((row or {}).get("risk_delta", 0.0) or 0.0)
            risk_acc = max(0.0, min(1.0, risk_acc + risk_delta))
            steps.append(
                {
                    "step_index": int((row or {}).get("step_index", idx) or idx),
                    "timestamp": str((row or {}).get("timestamp", incident.get("timestamp", _utc_now_iso()))),
                    "input": {
                        "type": str((row or {}).get("source_type", incident.get("source_type", "tool_output"))),
                        "source_hash": source_hash,
                        "content_redacted": snippet,
                        "trust_level": _derive_trust_level(str((row or {}).get("source_type", ""))),
                    },
                    "context_state_before": {
                        "active_policies": [policy_triggered] if policy_triggered and policy_triggered != "none" else [],
                        "memory_keys": [],
                        "retrieved_chunk_hashes": [source_hash] if source_hash else [],
                        "session_risk_accumulated": max(0.0, min(1.0, risk_acc - risk_delta)),
                    },
                    "model_decision": {
                        "output_redacted": snippet,
                        "tool_calls_attempted": (
                            [{"name": "tool_call", "params_redacted": payload}] if payload else []
                        ),
                        "reasoning_trace_available": False,
                    },
                    "security_evaluation": {
                        "risk_delta": risk_delta,
                        "policies_evaluated": [policy_triggered] if policy_triggered and policy_triggered != "none" else [],
                        "threshold_crossed": bool(idx >= int(threshold.get("triggered_at_step", 1) or 1)),
                        "action_taken": str(resolution.get("action", incident.get("action_taken", "allow_with_flag"))),
                        "reason_code": str(resolution.get("reason", incident.get("status", "logged_only"))),
                    },
                    "context_state_after": {
                        "memory_state_change": str((row or {}).get("memory_state_change", "")),
                        "new_context_hashes": [source_hash] if source_hash else [],
                        "session_risk_accumulated": risk_acc,
                    },
                    "provenance": {
                        "origin_session_id": str(incident.get("session_id", "")),
                        "cross_session_link": False,
                        "authorization_source": (
                            "system_config" if bool(incident.get("provenance_verified", False)) else "user_document"
                        ),
                    },
                }
            )
        timestamp = str(incident.get("timestamp", _utc_now_iso()))
        try:
            date_part = _parse_iso_utc(timestamp).strftime("%Y%m%d")
        except Exception:
            date_part = _utc_now().strftime("%Y%m%d")
        replay_id = f"rpl_{date_part}_{_uuid7_str().replace('-', '')[:10]}"
        return {
            "replay_id": replay_id,
            "incident_id": str(incident.get("incident_id", "")),
            "generated_at": _utc_now_iso(),
            "schema_version": "1.0.0",
            "environment_snapshot": {
                "agent_id": str(incident.get("agent_id", "")),
                "framework": "omega-walls",
                "model_provider": "unknown",
                "model_version": _infer_model_version({}),
                "policy_bundle_version": str(policy_triggered or "unknown"),
                "tool_gateway_config_hash": f"sha256:{sha256(policy_triggered.encode('utf-8')).hexdigest()}",
                "deployment_mode": "sidecar",
                "environment": str(incident.get("environment", "staging")),
            },
            "watermark": {
                "tenant_id": (str(incident.get("tenant_id", "")) or "unknown"),
                "api_key_hash": f"sha256:{sha256(str(job.get('requested_by_key_id', '')).encode('utf-8')).hexdigest()}",
                "export_scope": "raw" if include_raw else "redacted",
            },
            "steps": steps,
            "reproduction_instructions": {
                "runner_compatibility": "omega-walls >= 0.1.4",
                "mock_requirements": {
                    "tools_to_stub": [],
                    "retriever_mode": "hash_based_fixture",
                    "memory_backend": "ephemeral_dict",
                },
                "execution_command": "omega-walls replay run --manifest replay_pkg.json --sandbox",
                "expected_outcome": str(incident.get("status", "logged_only")),
                "notes": "Replay manifest generated from incident export chain and policy evidence.",
            },
        }

    def decrypt_package(self, *, package_path: str) -> Dict[str, Any]:
        path = Path(str(package_path))
        payload = path.read_bytes()
        plain = self.crypto.decrypt(payload)
        obj = json.loads(plain.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("invalid_manifest_payload")
        return obj

    def incident_retention_status(self, *, incident_id: str) -> str:
        row = self.incident_store.get_incident_any_age(incident_id=str(incident_id))
        if row is None:
            return "missing"
        cutoff = _retention_cutoff_iso(self.incident_retention_days)
        ts = str(row.get("timestamp", ""))
        if ts and ts < cutoff:
            return "expired"
        return "active"
