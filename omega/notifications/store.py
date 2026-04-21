"""Approval stores for notification callbacks and human decisions."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import threading
from typing import Dict, List, Optional

from omega.notifications.interfaces import ApprovalStore
from omega.notifications.models import ApprovalDecision, ApprovalRecord


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


class InMemoryApprovalStore(ApprovalStore):
    def __init__(self) -> None:
        self._rows: Dict[str, ApprovalRecord] = {}
        self._lock = threading.Lock()

    def create(self, record: ApprovalRecord) -> ApprovalRecord:
        with self._lock:
            self._rows[str(record.approval_id)] = record
            return record

    def get(self, approval_id: str) -> Optional[ApprovalRecord]:
        with self._lock:
            return self._rows.get(str(approval_id))

    def get_latest_for_session(self, *, tenant_id: str, session_id: str) -> Optional[ApprovalRecord]:
        with self._lock:
            candidates = [
                r
                for r in self._rows.values()
                if str(r.tenant_id) == str(tenant_id) and str(r.session_id) == str(session_id)
            ]
        if not candidates:
            return None
        candidates.sort(key=lambda x: str(x.created_at), reverse=True)
        return candidates[0]

    def resolve(self, approval_id: str, decision: ApprovalDecision) -> Optional[ApprovalRecord]:
        normalized = decision.normalized()
        with self._lock:
            row = self._rows.get(str(approval_id))
            if row is None:
                return None
            if str(row.status) != "pending":
                return row
            updated = replace(
                row,
                status=str(normalized.decision),
                updated_at=str(normalized.resolved_at),
                resolution=normalized,
            )
            self._rows[str(approval_id)] = updated
            return updated

    def mark_callback_id(self, approval_id: str, channel: str, callback_id: str) -> Optional[ApprovalRecord]:
        with self._lock:
            row = self._rows.get(str(approval_id))
            if row is None:
                return None
            callback_ids = dict(row.callback_ids)
            callback_ids[str(channel)] = str(callback_id)
            channels = sorted(set(list(row.channels) + [str(channel)]))
            updated = replace(row, callback_ids=callback_ids, channels=channels)
            self._rows[str(approval_id)] = updated
            return updated

    def expire_pending(self, *, now_iso: str) -> List[ApprovalRecord]:
        now_dt = _parse_iso(now_iso)
        out: List[ApprovalRecord] = []
        with self._lock:
            for approval_id, row in list(self._rows.items()):
                if str(row.status) != "pending":
                    continue
                if _parse_iso(row.expires_at) > now_dt:
                    continue
                decision = ApprovalDecision(decision="expired", source="timeout", resolved_at=now_iso)
                updated = replace(row, status="expired", updated_at=str(now_iso), resolution=decision)
                self._rows[str(approval_id)] = updated
                out.append(updated)
        return out

    def clear_session(self, *, tenant_id: str, session_id: str) -> int:
        deleted = 0
        with self._lock:
            for approval_id, row in list(self._rows.items()):
                if str(row.tenant_id) == str(tenant_id) and str(row.session_id) == str(session_id):
                    del self._rows[approval_id]
                    deleted += 1
        return deleted

    def close(self) -> None:
        return None


class SQLiteApprovalStore(ApprovalStore):
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
                CREATE TABLE IF NOT EXISTS approvals (
                  approval_id TEXT PRIMARY KEY,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  required_action TEXT NOT NULL,
                  tenant_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  trace_id TEXT NOT NULL,
                  decision_id TEXT NOT NULL,
                  control_outcome TEXT NOT NULL,
                  channels_json TEXT NOT NULL,
                  callback_ids_json TEXT NOT NULL,
                  resolution_json TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_approvals_session ON approvals(tenant_id, session_id, created_at DESC);
                CREATE INDEX IF NOT EXISTS idx_approvals_pending ON approvals(status, expires_at);
                """
            )

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ApprovalRecord:
        resolution_raw = row["resolution_json"]
        resolution = None
        if resolution_raw:
            data = json.loads(str(resolution_raw))
            resolution = ApprovalDecision(
                decision=str(data.get("decision", "")),
                actor_id=str(data.get("actor_id", "")),
                source=str(data.get("source", "")),
                reason=str(data.get("reason", "")),
                resolved_at=str(data.get("resolved_at", "")),
            )
        return ApprovalRecord(
            approval_id=str(row["approval_id"]),
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            expires_at=str(row["expires_at"]),
            required_action=str(row["required_action"]),
            tenant_id=str(row["tenant_id"]),
            session_id=str(row["session_id"]),
            actor_id=str(row["actor_id"]),
            trace_id=str(row["trace_id"]),
            decision_id=str(row["decision_id"]),
            control_outcome=str(row["control_outcome"]),
            channels=list(json.loads(str(row["channels_json"]))),
            callback_ids=dict(json.loads(str(row["callback_ids_json"]))),
            resolution=resolution,
        )

    @staticmethod
    def _insert_or_update(conn: sqlite3.Connection, record: ApprovalRecord) -> None:
        resolution_json = json.dumps(record.resolution.__dict__, ensure_ascii=False) if record.resolution else None
        conn.execute(
            """
            INSERT INTO approvals(
              approval_id, status, created_at, updated_at, expires_at, required_action,
              tenant_id, session_id, actor_id, trace_id, decision_id, control_outcome,
              channels_json, callback_ids_json, resolution_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(approval_id) DO UPDATE SET
              status=excluded.status,
              updated_at=excluded.updated_at,
              channels_json=excluded.channels_json,
              callback_ids_json=excluded.callback_ids_json,
              resolution_json=excluded.resolution_json
            """,
            (
                record.approval_id,
                record.status,
                record.created_at,
                record.updated_at,
                record.expires_at,
                record.required_action,
                record.tenant_id,
                record.session_id,
                record.actor_id,
                record.trace_id,
                record.decision_id,
                record.control_outcome,
                json.dumps(list(record.channels), ensure_ascii=False),
                json.dumps(dict(record.callback_ids), ensure_ascii=False),
                resolution_json,
            ),
        )

    def create(self, record: ApprovalRecord) -> ApprovalRecord:
        with self._lock, self._connect() as conn:
            self._insert_or_update(conn, record)
        return record

    def get(self, approval_id: str) -> Optional[ApprovalRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE approval_id = ?", (str(approval_id),)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_latest_for_session(self, *, tenant_id: str, session_id: str) -> Optional[ApprovalRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM approvals
                WHERE tenant_id = ? AND session_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (str(tenant_id), str(session_id)),
            ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def resolve(self, approval_id: str, decision: ApprovalDecision) -> Optional[ApprovalRecord]:
        normalized = decision.normalized()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE approval_id = ?", (str(approval_id),)).fetchone()
            if row is None:
                return None
            record = self._row_to_record(row)
            if str(record.status) != "pending":
                return record
            updated = replace(
                record,
                status=str(normalized.decision),
                updated_at=str(normalized.resolved_at),
                resolution=normalized,
            )
            self._insert_or_update(conn, updated)
        return updated

    def mark_callback_id(self, approval_id: str, channel: str, callback_id: str) -> Optional[ApprovalRecord]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE approval_id = ?", (str(approval_id),)).fetchone()
            if row is None:
                return None
            record = self._row_to_record(row)
            callback_ids = dict(record.callback_ids)
            callback_ids[str(channel)] = str(callback_id)
            channels = sorted(set(list(record.channels) + [str(channel)]))
            updated = replace(record, callback_ids=callback_ids, channels=channels)
            self._insert_or_update(conn, updated)
        return updated

    def expire_pending(self, *, now_iso: str) -> List[ApprovalRecord]:
        now_dt = _parse_iso(now_iso)
        expired: List[ApprovalRecord] = []
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE status = 'pending'",
            ).fetchall()
            for row in rows:
                record = self._row_to_record(row)
                if _parse_iso(record.expires_at) > now_dt:
                    continue
                decision = ApprovalDecision(decision="expired", source="timeout", resolved_at=now_iso)
                updated = replace(record, status="expired", updated_at=str(now_iso), resolution=decision)
                self._insert_or_update(conn, updated)
                expired.append(updated)
        return expired

    def clear_session(self, *, tenant_id: str, session_id: str) -> int:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM approvals WHERE tenant_id = ? AND session_id = ?",
                (str(tenant_id), str(session_id)),
            )
            return int(cur.rowcount or 0)

    def close(self) -> None:
        return None

