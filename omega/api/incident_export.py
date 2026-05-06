from __future__ import annotations

import base64
import json
import os
import re
import secrets
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from omega.telemetry.redaction import redact_text


_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_TOKEN_RE = re.compile(r"(?i)(api[_-]?key|token|password)\s*[:=]\s*[^\s]+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_iso_utc(value: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty timestamp")
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def _encode_cursor(*, ts: str, incident_id: str) -> str:
    payload = json.dumps({"ts": ts, "incident_id": incident_id}, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> Tuple[str, str]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("empty cursor")
    pad = "=" * ((4 - (len(raw) % 4)) % 4)
    payload = base64.urlsafe_b64decode((raw + pad).encode("ascii")).decode("utf-8")
    obj = json.loads(payload)
    if not isinstance(obj, dict):
        raise ValueError("invalid cursor")
    ts = str(obj.get("ts", "")).strip()
    incident_id = str(obj.get("incident_id", "")).strip()
    if not ts or not incident_id:
        raise ValueError("invalid cursor")
    _parse_iso_utc(ts)
    return ts, incident_id


def _uuid7_str() -> str:
    candidate = getattr(uuid, "uuid7", None)
    if callable(candidate):
        return str(candidate())
    # Fallback for runtimes where uuid7 is unavailable.
    return str(uuid.uuid4())


def _human_readable_id(dt: datetime, index: int) -> str:
    suffix = f"{int(index):04X}"[-4:]
    return f"INC-{dt.strftime('%Y%m%d')}-{suffix}"


def _split_csv_filter(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    out = [part.strip() for part in str(value).split(",") if part.strip()]
    return sorted(set(out))


def _severity_from_native(value: str, *, risk_score: float) -> str:
    raw = str(value or "").strip().upper()
    if raw == "L3":
        return "high" if risk_score < 0.95 else "critical"
    if raw == "L2":
        return "medium"
    if raw == "L1":
        return "low"
    return "high" if risk_score >= 0.75 else "medium"


def _status_from_control_outcome(control_outcome: str, action_types: Sequence[str]) -> str:
    actions = {str(x).strip().upper() for x in list(action_types) if str(x).strip()}
    outcome = str(control_outcome or "").strip().upper()
    if "TOOL_FREEZE" in actions:
        return "frozen"
    if "SOURCE_QUARANTINE" in actions:
        return "quarantined"
    if "HUMAN_ESCALATE" in actions or "REQUIRE_APPROVAL" in actions:
        return "escalated"
    if "SOFT_BLOCK" in actions or outcome in {"SOFT_BLOCK", "BLOCK"}:
        return "blocked"
    return "logged_only"


def _action_taken_from_status(status: str) -> str:
    if status == "blocked":
        return "block"
    if status == "frozen":
        return "freeze"
    if status == "quarantined":
        return "quarantine"
    return "allow_with_flag"


def _attack_type_from_walls(walls: Sequence[str]) -> str:
    wall_set = {str(x).strip().lower() for x in list(walls) if str(x).strip()}
    if "secret_exfiltration" in wall_set:
        return "data_exfil"
    if "tool_or_action_abuse" in wall_set:
        return "tool_abuse"
    if "policy_evasion" in wall_set:
        return "policy_drift"
    if "override_instructions" in wall_set:
        return "prompt_injection"
    return "unknown"


def _source_type_from_mime(mime: str) -> str:
    low = str(mime or "").strip().lower()
    if "pdf" in low:
        return "pdf"
    if "html" in low:
        return "web"
    if "email" in low:
        return "email"
    if "json" in low:
        return "api_response"
    return "tool_output"


def _summary_text(*, control_outcome: str, attack_type: str, severity: str, max_len: int = 120) -> str:
    text = f"{severity} incident: {attack_type} -> {control_outcome}".strip()
    return text[:max_len]


def _redacted_chain_from_payload(payload: Mapping[str, Any]) -> List[Dict[str, Any]]:
    monitor = payload.get("monitor", {}) if isinstance(payload.get("monitor", {}), Mapping) else {}
    fragments = monitor.get("fragments", []) if isinstance(monitor.get("fragments", []), list) else []
    rows: List[Dict[str, Any]] = []
    for idx, frag in enumerate(fragments, start=1):
        if not isinstance(frag, Mapping):
            continue
        red = redact_text(str(frag.get("excerpt_redacted", "")), max_chars=320)
        sanitized = _EMAIL_RE.sub("<REDACTED>", str(red.redacted))
        sanitized = _TOKEN_RE.sub(r"\1=<REDACTED>", sanitized)
        rows.append(
            {
                "step_index": idx,
                "timestamp": str(payload.get("ts", _utc_now_iso())),
                "source_type": str(frag.get("source_type", "tool_output") or "tool_output"),
                "source_hash": str(frag.get("excerpt_sha256", "")),
                "risk_delta": float(frag.get("contribution", 0.0) or 0.0),
                "snippet_redacted": str(sanitized),
                "tool_call_payload_redacted": "",
                "memory_state_change": "",
            }
        )
    return rows


class IncidentLightItem(BaseModel):
    incident_id: str
    human_readable_id: str
    timestamp: str
    severity: str = Field(pattern="^(low|medium|high|critical)$")
    status: str = Field(pattern="^(blocked|quarantined|frozen|escalated|logged_only)$")
    attack_type: str = Field(
        pattern="^(prompt_injection|policy_drift|tool_abuse|data_exfil|cross_session_accumulation|unknown)$"
    )
    agent_id: str
    environment: str = Field(pattern="^(dev|staging|prod)$")
    risk_score: float = Field(ge=0.0, le=1.0)
    source_type: str = Field(pattern="^(email|pdf|web|tool_output|memory_carryover|api_response)$")
    action_taken: str = Field(pattern="^(block|freeze|quarantine|allow_with_flag)$")
    steps_count: int = Field(ge=0)
    summary: str


class IncidentHeavyItem(IncidentLightItem):
    session_id: str
    tenant_id: Optional[str] = None
    chain_of_events: List[Dict[str, Any]]
    policy_triggered: str
    threshold_config: Dict[str, Any]
    resolution: Dict[str, Any]
    provenance_verified: bool


class PaginationEnvelope(BaseModel):
    has_more: bool
    next_cursor: Optional[str] = None
    total_in_range: int


class IncidentListResponse(BaseModel):
    data: List[IncidentLightItem]
    pagination: PaginationEnvelope


@dataclass(frozen=True)
class IncidentExportConfig:
    enabled: bool
    contract_version: str
    store_path: str
    key_store_path: str
    retention_days: int
    default_env: str
    rate_limit_rpm: int
    rate_limit_burst: int
    cors_allowed_origins: List[str]

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "IncidentExportConfig":
        data = dict(cfg or {})
        store_cfg = data.get("store", {}) if isinstance(data.get("store", {}), Mapping) else {}
        auth_cfg = data.get("auth", {}) if isinstance(data.get("auth", {}), Mapping) else {}
        rate_cfg = data.get("rate_limit", {}) if isinstance(data.get("rate_limit", {}), Mapping) else {}
        cors_cfg = data.get("cors", {}) if isinstance(data.get("cors", {}), Mapping) else {}
        origins = cors_cfg.get("allowed_origins", [])
        if not isinstance(origins, list):
            origins = []
        return cls(
            enabled=bool(data.get("enabled", False)),
            contract_version=str(data.get("contract_version", "1.0")).strip() or "1.0",
            store_path=str(store_cfg.get("sqlite_path", "artifacts/state/incident_export.db")).strip()
            or "artifacts/state/incident_export.db",
            key_store_path=str(auth_cfg.get("key_store_path", "artifacts/state/incident_export_keys.db")).strip()
            or "artifacts/state/incident_export_keys.db",
            retention_days=max(1, int(data.get("retention_days", 30))),
            default_env=str(data.get("default_environment", "staging")).strip().lower() or "staging",
            rate_limit_rpm=max(1, int(rate_cfg.get("rpm", 60))),
            rate_limit_burst=max(1, int(rate_cfg.get("burst", 10))),
            cors_allowed_origins=[str(x).strip() for x in origins if str(x).strip()],
        )


class IncidentExportStore:
    def __init__(self, *, sqlite_path: str | Path, retention_days: int = 30) -> None:
        self.sqlite_path = Path(str(sqlite_path))
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_days = max(1, int(retention_days))
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
                CREATE TABLE IF NOT EXISTS incidents (
                  incident_id TEXT PRIMARY KEY,
                  human_readable_id TEXT NOT NULL,
                  ts TEXT NOT NULL,
                  severity TEXT NOT NULL,
                  status TEXT NOT NULL,
                  attack_type TEXT NOT NULL,
                  agent_id TEXT NOT NULL,
                  environment TEXT NOT NULL,
                  risk_score REAL NOT NULL,
                  source_type TEXT NOT NULL,
                  action_taken TEXT NOT NULL,
                  steps_count INTEGER NOT NULL,
                  summary TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  tenant_id TEXT,
                  chain_of_events_json TEXT NOT NULL,
                  policy_triggered TEXT NOT NULL,
                  threshold_config_json TEXT NOT NULL,
                  resolution_json TEXT NOT NULL,
                  provenance_verified INTEGER NOT NULL,
                  created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_incidents_ts_id ON incidents(ts DESC, incident_id DESC);
                CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
                CREATE INDEX IF NOT EXISTS idx_incidents_status ON incidents(status);
                CREATE INDEX IF NOT EXISTS idx_incidents_environment ON incidents(environment);
                CREATE INDEX IF NOT EXISTS idx_incidents_agent_id ON incidents(agent_id);
                """
            )

    def _retention_cutoff_iso(self) -> str:
        dt = _utc_now() - timedelta(days=int(self.retention_days))
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    def health_snapshot(self) -> Dict[str, Any]:
        exists = self.sqlite_path.exists()
        size = int(self.sqlite_path.stat().st_size) if exists else 0
        return {
            "enabled": True,
            "sqlite_path": str(self.sqlite_path),
            "exists": bool(exists),
            "size_bytes": size,
            "retention_days": int(self.retention_days),
            "retention_cutoff": self._retention_cutoff_iso(),
        }

    def insert_record(self, record: Mapping[str, Any]) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO incidents(
                  incident_id, human_readable_id, ts, severity, status, attack_type, agent_id,
                  environment, risk_score, source_type, action_taken, steps_count, summary,
                  session_id, tenant_id, chain_of_events_json, policy_triggered, threshold_config_json,
                  resolution_json, provenance_verified, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.get("incident_id", "")),
                    str(record.get("human_readable_id", "")),
                    str(record.get("timestamp", "")),
                    str(record.get("severity", "")),
                    str(record.get("status", "")),
                    str(record.get("attack_type", "")),
                    str(record.get("agent_id", "")),
                    str(record.get("environment", "")),
                    float(record.get("risk_score", 0.0)),
                    str(record.get("source_type", "")),
                    str(record.get("action_taken", "")),
                    int(record.get("steps_count", 0)),
                    str(record.get("summary", "")),
                    str(record.get("session_id", "")),
                    (str(record.get("tenant_id", "")) or None),
                    json.dumps(list(record.get("chain_of_events", []) or []), ensure_ascii=False),
                    str(record.get("policy_triggered", "")),
                    json.dumps(dict(record.get("threshold_config", {}) or {}), ensure_ascii=False),
                    json.dumps(dict(record.get("resolution", {}) or {}), ensure_ascii=False),
                    1 if bool(record.get("provenance_verified", False)) else 0,
                    _utc_now_iso(),
                ),
            )

    def _where_filters(
        self,
        *,
        after: Optional[str],
        before: Optional[str],
        cursor: Optional[str],
        filters: Mapping[str, str],
    ) -> Tuple[str, List[Any]]:
        clauses: List[str] = ["ts >= ?"]
        params: List[Any] = [self._retention_cutoff_iso()]

        if after:
            _ = _parse_iso_utc(after)
            clauses.append("ts >= ?")
            params.append(str(after))
        if before:
            _ = _parse_iso_utc(before)
            clauses.append("ts <= ?")
            params.append(str(before))

        if cursor:
            c_ts, c_id = _decode_cursor(cursor)
            clauses.append("(ts < ? OR (ts = ? AND incident_id < ?))")
            params.extend([c_ts, c_ts, c_id])

        for column, raw in filters.items():
            values = _split_csv_filter(raw)
            if not values:
                continue
            placeholders = ",".join(["?"] * len(values))
            clauses.append(f"{column} IN ({placeholders})")
            params.extend(values)

        return " AND ".join(clauses), params

    def list_incidents(
        self,
        *,
        limit: int,
        cursor: Optional[str],
        after: Optional[str],
        before: Optional[str],
        filters: Mapping[str, str],
    ) -> Tuple[List[Dict[str, Any]], bool, Optional[str], int]:
        safe_limit = max(1, min(200, int(limit)))
        where_sql, params = self._where_filters(after=after, before=before, cursor=cursor, filters=filters)
        with self._lock, self._connect() as conn:
            total = int(conn.execute(f"SELECT COUNT(1) AS cnt FROM incidents WHERE {where_sql}", tuple(params)).fetchone()["cnt"])
            rows = conn.execute(
                f"""
                SELECT
                  incident_id, human_readable_id, ts, severity, status, attack_type, agent_id,
                  environment, risk_score, source_type, action_taken, steps_count, summary
                FROM incidents
                WHERE {where_sql}
                ORDER BY ts DESC, incident_id DESC
                LIMIT ?
                """,
                tuple(params + [safe_limit + 1]),
            ).fetchall()

        has_more = len(rows) > safe_limit
        rows = rows[:safe_limit]
        data: List[Dict[str, Any]] = []
        for row in rows:
            item = {
                "incident_id": str(row["incident_id"]),
                "human_readable_id": str(row["human_readable_id"]),
                "timestamp": str(row["ts"]),
                "severity": str(row["severity"]),
                "status": str(row["status"]),
                "attack_type": str(row["attack_type"]),
                "agent_id": str(row["agent_id"]),
                "environment": str(row["environment"]),
                "risk_score": float(row["risk_score"]),
                "source_type": str(row["source_type"]),
                "action_taken": str(row["action_taken"]),
                "steps_count": int(row["steps_count"]),
                "summary": str(row["summary"]),
            }
            data.append(item)

        next_cursor = None
        if has_more and rows:
            last = rows[-1]
            next_cursor = _encode_cursor(ts=str(last["ts"]), incident_id=str(last["incident_id"]))
        return data, has_more, next_cursor, total

    def get_incident(self, *, incident_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  incident_id, human_readable_id, ts, severity, status, attack_type, agent_id,
                  environment, risk_score, source_type, action_taken, steps_count, summary,
                  session_id, tenant_id, chain_of_events_json, policy_triggered,
                  threshold_config_json, resolution_json, provenance_verified
                FROM incidents
                WHERE incident_id = ? AND ts >= ?
                """,
                (str(incident_id), self._retention_cutoff_iso()),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_incident(row)

    def get_incident_any_age(self, *, incident_id: str) -> Optional[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                  incident_id, human_readable_id, ts, severity, status, attack_type, agent_id,
                  environment, risk_score, source_type, action_taken, steps_count, summary,
                  session_id, tenant_id, chain_of_events_json, policy_triggered,
                  threshold_config_json, resolution_json, provenance_verified
                FROM incidents
                WHERE incident_id = ?
                """,
                (str(incident_id),),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_incident(row)

    def _row_to_incident(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {
            "incident_id": str(row["incident_id"]),
            "human_readable_id": str(row["human_readable_id"]),
            "timestamp": str(row["ts"]),
            "severity": str(row["severity"]),
            "status": str(row["status"]),
            "attack_type": str(row["attack_type"]),
            "agent_id": str(row["agent_id"]),
            "environment": str(row["environment"]),
            "risk_score": float(row["risk_score"]),
            "source_type": str(row["source_type"]),
            "action_taken": str(row["action_taken"]),
            "steps_count": int(row["steps_count"]),
            "summary": str(row["summary"]),
            "session_id": str(row["session_id"]),
            "tenant_id": (str(row["tenant_id"]) if row["tenant_id"] is not None else None),
            "chain_of_events": list(json.loads(str(row["chain_of_events_json"]) or "[]")),
            "policy_triggered": str(row["policy_triggered"]),
            "threshold_config": dict(json.loads(str(row["threshold_config_json"]) or "{}")),
            "resolution": dict(json.loads(str(row["resolution_json"]) or "{}")),
            "provenance_verified": bool(int(row["provenance_verified"])),
        }


class IncidentRateLimiter:
    def __init__(self, *, rpm: int, burst: int) -> None:
        self.rpm = max(1, int(rpm))
        self.burst = max(1, int(burst))
        self.refill_per_sec = float(self.rpm) / 60.0
        self._state: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def check(self, *, key_ref: str) -> Tuple[bool, int, int]:
        now = time.time()
        with self._lock:
            row = self._state.get(key_ref)
            if row is None:
                row = {"tokens": float(self.burst), "last": now}
                self._state[key_ref] = row
            elapsed = max(0.0, now - float(row["last"]))
            row["tokens"] = min(float(self.burst), float(row["tokens"]) + elapsed * self.refill_per_sec)
            row["last"] = now
            allowed = row["tokens"] >= 1.0
            if allowed:
                row["tokens"] -= 1.0
            remaining = max(0, int(row["tokens"]))
            need = max(0.0, 1.0 - float(row["tokens"]))
            reset_sec = int(now + (need / self.refill_per_sec if self.refill_per_sec > 0 else 1.0))
            return bool(allowed), remaining, reset_sec


@dataclass
class IncidentApiKeyRecord:
    key_id: str
    key_hash: str
    scopes: List[str]
    status: str
    created_at: str
    rotated_from: Optional[str]
    revoked_at: Optional[str]


class IncidentApiKeyStore:
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
                CREATE TABLE IF NOT EXISTS incident_api_keys (
                  key_id TEXT PRIMARY KEY,
                  key_hash TEXT NOT NULL,
                  scopes_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  rotated_from TEXT,
                  revoked_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_incident_api_keys_status ON incident_api_keys(status);
                """
            )

    def _hash_key(self, secret: str) -> str:
        try:
            import bcrypt

            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(secret.encode("utf-8"), salt).decode("utf-8")
            return f"bcrypt:{hashed}"
        except Exception:
            return f"sha256:{sha256(secret.encode('utf-8')).hexdigest()}"

    def _verify_hash(self, *, secret: str, key_hash: str) -> bool:
        text = str(key_hash or "").strip()
        if text.startswith("bcrypt:"):
            try:
                import bcrypt

                return bool(bcrypt.checkpw(secret.encode("utf-8"), text.split(":", 1)[1].encode("utf-8")))
            except Exception:
                return False
        if text.startswith("sha256:"):
            return secrets.compare_digest(text.split(":", 1)[1], sha256(secret.encode("utf-8")).hexdigest())
        return False

    def create_key(self, *, scopes: Sequence[str], rotated_from: Optional[str] = None) -> Dict[str, Any]:
        raw = f"owx_{secrets.token_urlsafe(36)}"
        key_id = _uuid7_str()
        now = _utc_now_iso()
        record = IncidentApiKeyRecord(
            key_id=key_id,
            key_hash=self._hash_key(raw),
            scopes=sorted(set(str(x).strip() for x in list(scopes) if str(x).strip())),
            status="active",
            created_at=now,
            rotated_from=(str(rotated_from).strip() or None),
            revoked_at=None,
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO incident_api_keys(key_id, key_hash, scopes_json, status, created_at, rotated_from, revoked_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.key_id,
                    record.key_hash,
                    json.dumps(record.scopes, ensure_ascii=True),
                    record.status,
                    record.created_at,
                    record.rotated_from,
                    record.revoked_at,
                ),
            )
        return {
            "key_id": record.key_id,
            "api_key": raw,
            "scopes": list(record.scopes),
            "created_at": record.created_at,
            "status": record.status,
        }

    def revoke(self, *, key_id: str) -> bool:
        now = _utc_now_iso()
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE incident_api_keys SET status='revoked', revoked_at=? WHERE key_id=? AND status='active'",
                (now, str(key_id)),
            )
            return int(cur.rowcount or 0) > 0

    def rotate(self, *, key_id: str, scopes: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
        target = self.get(key_id=key_id)
        if target is None:
            return None
        if str(target.status) != "active":
            return None
        selected_scopes = list(scopes) if scopes is not None else list(target.scopes)
        if not self.revoke(key_id=key_id):
            return None
        return self.create_key(scopes=selected_scopes, rotated_from=key_id)

    def get(self, *, key_id: str) -> Optional[IncidentApiKeyRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM incident_api_keys WHERE key_id=?", (str(key_id),)).fetchone()
        if row is None:
            return None
        return IncidentApiKeyRecord(
            key_id=str(row["key_id"]),
            key_hash=str(row["key_hash"]),
            scopes=list(json.loads(str(row["scopes_json"]) or "[]")),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            rotated_from=(str(row["rotated_from"]) if row["rotated_from"] is not None else None),
            revoked_at=(str(row["revoked_at"]) if row["revoked_at"] is not None else None),
        )

    def list_keys(self) -> List[Dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT key_id, scopes_json, status, created_at, rotated_from, revoked_at FROM incident_api_keys ORDER BY created_at DESC"
            ).fetchall()
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "key_id": str(row["key_id"]),
                    "scopes": list(json.loads(str(row["scopes_json"]) or "[]")),
                    "status": str(row["status"]),
                    "created_at": str(row["created_at"]),
                    "rotated_from": (str(row["rotated_from"]) if row["rotated_from"] is not None else None),
                    "revoked_at": (str(row["revoked_at"]) if row["revoked_at"] is not None else None),
                }
            )
        return out

    def resolve_key(self, *, provided_key: str) -> Optional[IncidentApiKeyRecord]:
        secret = str(provided_key or "").strip()
        if not secret:
            return None
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT * FROM incident_api_keys WHERE status='active'").fetchall()
        for row in rows:
            key_hash = str(row["key_hash"])
            if self._verify_hash(secret=secret, key_hash=key_hash):
                return IncidentApiKeyRecord(
                    key_id=str(row["key_id"]),
                    key_hash=key_hash,
                    scopes=list(json.loads(str(row["scopes_json"]) or "[]")),
                    status=str(row["status"]),
                    created_at=str(row["created_at"]),
                    rotated_from=(str(row["rotated_from"]) if row["rotated_from"] is not None else None),
                    revoked_at=(str(row["revoked_at"]) if row["revoked_at"] is not None else None),
                )
        return None


def build_incident_record_from_scan(
    *,
    payload: Mapping[str, Any],
    parsed: Mapping[str, Any],
    environment: str,
    runtime_mode: str,
) -> Dict[str, Any]:
    now = _utc_now()
    incident_id = _uuid7_str()
    step = int((((payload.get("policy_trace", {}) or {}).get("step", 1)) or 1))
    risk_score = float(payload.get("risk_score", 0.0) or 0.0)
    risk_norm = max(0.0, min(1.0, risk_score / 100.0 if risk_score > 1.0 else risk_score))
    policy_trace = payload.get("policy_trace", {}) if isinstance(payload.get("policy_trace", {}), Mapping) else {}
    walls = list(policy_trace.get("walls_triggered", []) or [])
    actions = list(policy_trace.get("action_types", []) or [])
    control_outcome = str(payload.get("control_outcome", "ALLOW"))
    status = _status_from_control_outcome(control_outcome, actions)
    attack_type = _attack_type_from_walls(walls)
    severity = _severity_from_native(str(policy_trace.get("severity", "L2")), risk_score=risk_norm)
    source_type = _source_type_from_mime(str(parsed.get("mime", "")))
    chain = _redacted_chain_from_payload(payload)
    return {
        "incident_id": incident_id,
        "human_readable_id": _human_readable_id(now, step),
        "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "severity": severity,
        "status": status,
        "attack_type": attack_type,
        "agent_id": str(parsed.get("actor_id") or parsed.get("session_id") or "agent:unknown"),
        "environment": str(environment or "staging"),
        "risk_score": risk_norm,
        "source_type": source_type,
        "action_taken": _action_taken_from_status(status),
        "steps_count": max(step, len(chain)),
        "summary": _summary_text(control_outcome=control_outcome, attack_type=attack_type, severity=severity),
        "session_id": str(parsed.get("session_id") or f"req:{payload.get('request_id', '')}"),
        "tenant_id": (str(parsed.get("tenant_id", "")) or None),
        "chain_of_events": chain,
        "policy_triggered": ",".join(sorted(set(str(x) for x in walls if str(x).strip()))) or "none",
        "threshold_config": {
            "base_threshold": float(policy_trace.get("sigma", 0.0) or 0.0),
            "accumulated_risk": risk_norm,
            "triggered_at_step": step,
            "runtime_mode": str(runtime_mode),
        },
        "resolution": {
            "action": _action_taken_from_status(status),
            "reason": str(control_outcome),
            "auto_remediation_applied": status in {"blocked", "quarantined", "frozen"},
            "manual_review_required": status in {"escalated"},
        },
        "provenance_verified": True,
    }


def incident_access_log_record(*, key_hash: str, endpoint: str, status_code: int, ip: str) -> Dict[str, Any]:
    return {
        "event": "incident_export_access",
        "ts": _utc_now_iso(),
        "key_hash": str(key_hash),
        "endpoint": str(endpoint),
        "status_code": int(status_code),
        "ip": str(ip),
    }


def key_fingerprint(value: str) -> str:
    return sha256(str(value or "").encode("utf-8")).hexdigest()
