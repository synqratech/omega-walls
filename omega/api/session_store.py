"""Session runtime persistence for API stateful mode."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import time
from typing import Any, Dict, Optional

import numpy as np

from omega.interfaces.contracts_v1 import K_V1
from omega.validation.numeric import validate_state_values


def _now_ts() -> int:
    return int(time.time())


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _strict_json_loads(value: str) -> Any:
    return json.loads(value, parse_constant=_reject_nonstandard_json_constant)


@dataclass(frozen=True)
class SessionStateRow:
    tenant_id: str
    session_id: str
    actor_id: str
    m: np.ndarray
    step: int
    updated_at_ts: int
    expires_at_ts: int


class ApiSessionStore:
    def __init__(
        self,
        *,
        sqlite_path: str | Path,
        session_ttl_sec: int = 86_400,
        request_cache_ttl_sec: int = 86_400,
    ) -> None:
        self.sqlite_path = Path(str(sqlite_path))
        self.session_ttl_sec = max(60, int(session_ttl_sec))
        self.request_cache_ttl_sec = max(60, int(request_cache_ttl_sec))
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
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
                CREATE TABLE IF NOT EXISTS session_state (
                  tenant_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  actor_id TEXT NOT NULL,
                  m_json TEXT NOT NULL,
                  step INTEGER NOT NULL,
                  updated_at_ts INTEGER NOT NULL,
                  expires_at_ts INTEGER NOT NULL,
                  PRIMARY KEY (tenant_id, session_id)
                );

                CREATE TABLE IF NOT EXISTS request_cache (
                  tenant_id TEXT NOT NULL,
                  session_id TEXT NOT NULL,
                  request_id TEXT NOT NULL,
                  response_json TEXT NOT NULL,
                  created_at_ts INTEGER NOT NULL,
                  expires_at_ts INTEGER NOT NULL,
                  PRIMARY KEY (tenant_id, session_id, request_id)
                );
                """
            )

    def _cleanup_expired_conn(self, conn: sqlite3.Connection, now_ts: int) -> None:
        conn.execute("DELETE FROM session_state WHERE expires_at_ts < ?", (int(now_ts),))
        conn.execute("DELETE FROM request_cache WHERE expires_at_ts < ?", (int(now_ts),))

    def load_session_state(self, *, tenant_id: str, session_id: str) -> Optional[SessionStateRow]:
        now_ts = _now_ts()
        with self._connect() as conn:
            self._cleanup_expired_conn(conn, now_ts)
            row = conn.execute(
                """
                SELECT tenant_id, session_id, actor_id, m_json, step, updated_at_ts, expires_at_ts
                FROM session_state
                WHERE tenant_id = ? AND session_id = ?
                """,
                (str(tenant_id), str(session_id)),
            ).fetchone()
            if row is None:
                return None
            try:
                parsed = _strict_json_loads(str(row["m_json"]))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError("corrupt session state: invalid strict JSON") from exc
            arr = validate_state_values(vector=parsed, wall_count=K_V1, name="persisted state.m")
            return SessionStateRow(
                tenant_id=str(row["tenant_id"]),
                session_id=str(row["session_id"]),
                actor_id=str(row["actor_id"]),
                m=arr,
                step=int(row["step"]),
                updated_at_ts=int(row["updated_at_ts"]),
                expires_at_ts=int(row["expires_at_ts"]),
            )

    def save_session_state(
        self,
        *,
        tenant_id: str,
        session_id: str,
        actor_id: str,
        m: np.ndarray,
        step: int,
    ) -> None:
        now_ts = _now_ts()
        expires = now_ts + int(self.session_ttl_sec)
        validated_m = validate_state_values(vector=m, wall_count=K_V1, name="state.m")
        if isinstance(step, bool) or int(step) != step or int(step) < 0:
            raise ValueError("state.step must be a nonnegative integer")
        payload = json.dumps(validated_m.tolist(), ensure_ascii=False, allow_nan=False)
        with self._connect() as conn:
            self._cleanup_expired_conn(conn, now_ts)
            conn.execute(
                """
                INSERT INTO session_state(tenant_id, session_id, actor_id, m_json, step, updated_at_ts, expires_at_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, session_id) DO UPDATE SET
                  actor_id = excluded.actor_id,
                  m_json = excluded.m_json,
                  step = excluded.step,
                  updated_at_ts = excluded.updated_at_ts,
                  expires_at_ts = excluded.expires_at_ts
                """,
                (
                    str(tenant_id),
                    str(session_id),
                    str(actor_id),
                    payload,
                    int(step),
                    int(now_ts),
                    int(expires),
                ),
            )

    def get_cached_response(self, *, tenant_id: str, session_id: str, request_id: str) -> Optional[Dict[str, Any]]:
        now_ts = _now_ts()
        with self._connect() as conn:
            self._cleanup_expired_conn(conn, now_ts)
            row = conn.execute(
                """
                SELECT response_json
                FROM request_cache
                WHERE tenant_id = ? AND session_id = ? AND request_id = ?
                """,
                (str(tenant_id), str(session_id), str(request_id)),
            ).fetchone()
            if row is None:
                return None
            try:
                payload = _strict_json_loads(str(row["response_json"]))
            except (json.JSONDecodeError, ValueError):
                return None
            if not isinstance(payload, dict):
                return None
            return payload

    def save_cached_response(
        self,
        *,
        tenant_id: str,
        session_id: str,
        request_id: str,
        response_payload: Dict[str, Any],
    ) -> None:
        now_ts = _now_ts()
        expires = now_ts + int(self.request_cache_ttl_sec)
        blob = json.dumps(dict(response_payload), ensure_ascii=False, allow_nan=False)
        with self._connect() as conn:
            self._cleanup_expired_conn(conn, now_ts)
            conn.execute(
                """
                INSERT INTO request_cache(tenant_id, session_id, request_id, response_json, created_at_ts, expires_at_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, session_id, request_id) DO UPDATE SET
                  response_json = excluded.response_json,
                  created_at_ts = excluded.created_at_ts,
                  expires_at_ts = excluded.expires_at_ts
                """,
                (
                    str(tenant_id),
                    str(session_id),
                    str(request_id),
                    blob,
                    int(now_ts),
                    int(expires),
                ),
            )

    def save_state_and_cached_response(
        self,
        *,
        tenant_id: str,
        session_id: str,
        actor_id: str,
        m: np.ndarray,
        step: int,
        request_id: str,
        response_payload: Dict[str, Any],
    ) -> None:
        """Persist state and idempotency cache in one SQLite transaction."""
        now_ts = _now_ts()
        state_expires = now_ts + int(self.session_ttl_sec)
        cache_expires = now_ts + int(self.request_cache_ttl_sec)
        validated_m = validate_state_values(vector=m, wall_count=K_V1, name="state.m")
        if isinstance(step, bool) or int(step) != step or int(step) < 0:
            raise ValueError("state.step must be a nonnegative integer")
        state_blob = json.dumps(validated_m.tolist(), ensure_ascii=False, allow_nan=False)
        response_blob = json.dumps(dict(response_payload), ensure_ascii=False, allow_nan=False)
        with self._connect() as conn:
            self._cleanup_expired_conn(conn, now_ts)
            conn.execute(
                """
                INSERT INTO session_state(tenant_id, session_id, actor_id, m_json, step, updated_at_ts, expires_at_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, session_id) DO UPDATE SET
                  actor_id = excluded.actor_id,
                  m_json = excluded.m_json,
                  step = excluded.step,
                  updated_at_ts = excluded.updated_at_ts,
                  expires_at_ts = excluded.expires_at_ts
                """,
                (
                    str(tenant_id),
                    str(session_id),
                    str(actor_id),
                    state_blob,
                    int(step),
                    int(now_ts),
                    int(state_expires),
                ),
            )
            conn.execute(
                """
                INSERT INTO request_cache(tenant_id, session_id, request_id, response_json, created_at_ts, expires_at_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(tenant_id, session_id, request_id) DO UPDATE SET
                  response_json = excluded.response_json,
                  created_at_ts = excluded.created_at_ts,
                  expires_at_ts = excluded.expires_at_ts
                """,
                (
                    str(tenant_id),
                    str(session_id),
                    str(request_id),
                    response_blob,
                    int(now_ts),
                    int(cache_expires),
                ),
            )

    def clear_session(self, *, tenant_id: str, session_id: str) -> bool:
        with self._connect() as conn:
            cur1 = conn.execute(
                "DELETE FROM session_state WHERE tenant_id = ? AND session_id = ?",
                (str(tenant_id), str(session_id)),
            )
            cur2 = conn.execute(
                "DELETE FROM request_cache WHERE tenant_id = ? AND session_id = ?",
                (str(tenant_id), str(session_id)),
            )
            return int(cur1.rowcount or 0) > 0 or int(cur2.rowcount or 0) > 0
