"""Cross-session persistent enforcement and risk state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, List, Optional, Tuple
import uuid

import numpy as np

from omega.interfaces.contracts_v1 import K_V1, OffAction

LOGGER = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_list(value: Optional[str], default: Iterable[Any]) -> List[Any]:
    if not value:
        return list(default)
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return list(default)
    if not isinstance(parsed, list):
        return list(default)
    return parsed


def _to_vec(values: Iterable[float], k: int) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size != k:
        out = np.zeros(k, dtype=float)
        out[: min(k, arr.size)] = arr[: min(k, arr.size)]
        return out
    return arr


@dataclass(frozen=True)
class HydratedState:
    actor_id: str
    actor_hash: str
    global_step: int
    carryover_applied: bool
    carried_scars_before_decay: np.ndarray
    carried_scars_after_decay: np.ndarray
    freeze_until_step: Optional[int]
    quarantined_sources_count: int
    state_backend: str = "sqlite"
    state_version: str = "1.0"


@dataclass
class CrossSessionStateManager:
    config: Dict[str, Any]
    enabled: bool
    sqlite_path: Path
    fallback_actor_to_session: bool
    actor_ttl_steps: int
    decay_mode: str
    decay_half_life_steps: float
    transfer_scars: bool
    transfer_freeze: bool
    transfer_quarantine: bool
    freeze_scope: str
    quarantine_scope: str
    strikes_to_quarantine: int
    source_quarantine_horizon_steps: int
    telemetry_hash_actor_id: bool
    telemetry_actor_hash_salt_env: str
    k: int

    def __post_init__(self) -> None:
        if self.enabled:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_db()
        self._last_hydrated: Dict[Tuple[str, str], HydratedState] = {}
        if self.decay_half_life_steps <= 0:
            raise ValueError("cross_session.decay.half_life_steps must be > 0")

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CrossSessionStateManager":
        off = config.get("off_policy", {})
        cs = off.get("cross_session", {})
        sq = off.get("source_quarantine", {})
        default_path = "artifacts/state/cross_session_state.db"
        sqlite_path = str(cs.get("sqlite_path", default_path))
        if os.getenv("PYTEST_CURRENT_TEST") and sqlite_path == default_path:
            sqlite_path = f"artifacts/state/pytest_cross_session_{os.getpid()}_{uuid.uuid4().hex}.db"

        if "duration_steps" in sq:
            horizon_steps = int(sq.get("duration_steps", 24))
        else:
            # Backward compatibility: existing v1 configs use duration_hours but runtime is step-based.
            horizon_steps = int(sq.get("duration_hours", 24))
            if "duration_hours" in sq:
                LOGGER.warning(
                    "off_policy.source_quarantine.duration_hours is interpreted as step horizon; "
                    "prefer duration_steps (deprecated field)."
                )

        return cls(
            config=config,
            enabled=bool(cs.get("enabled", False)),
            sqlite_path=Path(sqlite_path),
            fallback_actor_to_session=bool(cs.get("fallback_actor_to_session", True)),
            actor_ttl_steps=int(cs.get("actor_ttl_steps", 400)),
            decay_mode=str(cs.get("decay", {}).get("mode", "exponential")).lower(),
            decay_half_life_steps=float(cs.get("decay", {}).get("half_life_steps", 120)),
            transfer_scars=bool(cs.get("transfer", {}).get("scars", True)),
            transfer_freeze=bool(cs.get("transfer", {}).get("freeze", True)),
            transfer_quarantine=bool(cs.get("transfer", {}).get("quarantine", True)),
            freeze_scope=str(cs.get("freeze_scope", "actor")).lower(),
            quarantine_scope=str(cs.get("quarantine_scope", "global_source")).lower(),
            strikes_to_quarantine=int(sq.get("strikes_to_quarantine", 2)),
            source_quarantine_horizon_steps=horizon_steps,
            telemetry_hash_actor_id=bool(cs.get("telemetry", {}).get("hash_actor_id", True)),
            telemetry_actor_hash_salt_env=str(cs.get("telemetry", {}).get("actor_hash_salt_env", "OMEGA_ACTOR_HASH_SALT")),
            k=int(config.get("omega", {}).get("K", K_V1)),
        )

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
                CREATE TABLE IF NOT EXISTS meta (
                  key TEXT PRIMARY KEY,
                  value INTEGER NOT NULL
                );

                CREATE TABLE IF NOT EXISTS actor_state (
                  actor_id TEXT PRIMARY KEY,
                  scars_json TEXT NOT NULL,
                  last_step INTEGER NOT NULL,
                  expires_at_step INTEGER NOT NULL,
                  updated_at_ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS actor_freeze (
                  actor_id TEXT PRIMARY KEY,
                  freeze_until_step INTEGER NOT NULL,
                  tool_mode TEXT NOT NULL,
                  allowlist_json TEXT NOT NULL,
                  updated_at_ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS source_quarantine (
                  source_id TEXT PRIMARY KEY,
                  strikes INTEGER NOT NULL,
                  quarantined_until_step INTEGER NOT NULL,
                  updated_at_ts TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS session_actor (
                  session_id TEXT PRIMARY KEY,
                  actor_id TEXT NOT NULL,
                  created_at_ts TEXT NOT NULL,
                  updated_at_ts TEXT NOT NULL,
                  last_step INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_actor_state_expires ON actor_state (expires_at_step);
                CREATE INDEX IF NOT EXISTS idx_source_quarantine_until ON source_quarantine (quarantined_until_step);
                """
            )
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('global_step', 0)")

    def _next_global_step(self, conn: sqlite3.Connection) -> int:
        conn.execute("UPDATE meta SET value = value + 1 WHERE key = 'global_step'")
        row = conn.execute("SELECT value FROM meta WHERE key = 'global_step'").fetchone()
        return int(row["value"])

    def _current_global_step(self, conn: sqlite3.Connection) -> int:
        row = conn.execute("SELECT value FROM meta WHERE key = 'global_step'").fetchone()
        if row is None:
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('global_step', 0)")
            return 0
        return int(row["value"])

    def _actor_hash(self, actor_id: str) -> str:
        if not self.telemetry_hash_actor_id:
            return actor_id
        salt = os.getenv(self.telemetry_actor_hash_salt_env, "omega-default-salt")
        payload = f"{salt}:{actor_id}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _decay_scars(self, scars: np.ndarray, delta_steps: int) -> np.ndarray:
        if not self.transfer_scars:
            return np.zeros_like(scars)
        if self.decay_mode == "exponential":
            decay_k = math.log(2.0) / self.decay_half_life_steps
            factor = math.exp(-decay_k * max(0, int(delta_steps)))
            return scars * factor
        raise ValueError(f"Unsupported cross_session decay mode: {self.decay_mode}")

    def _resolve_run_step(self, conn: sqlite3.Connection, actor_id: str, session_id: str) -> int:
        row = conn.execute("SELECT last_step FROM session_actor WHERE session_id = ?", (session_id,)).fetchone()
        if row is not None:
            return int(row["last_step"])
        # Fallback to current global step when hydrate was not called.
        return self._current_global_step(conn)

    def hydrate_actor_state(self, actor_id: str, session_id: str) -> HydratedState:
        if not self.enabled:
            return HydratedState(
                actor_id=actor_id,
                actor_hash=self._actor_hash(actor_id),
                global_step=0,
                carryover_applied=False,
                carried_scars_before_decay=np.zeros(self.k, dtype=float),
                carried_scars_after_decay=np.zeros(self.k, dtype=float),
                freeze_until_step=None,
                quarantined_sources_count=0,
            )

        now = _utc_now_iso()
        with self._connect() as conn:
            step = self._next_global_step(conn)
            cur = conn.execute(
                "SELECT actor_id, created_at_ts FROM session_actor WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if cur is None:
                conn.execute(
                    """
                    INSERT INTO session_actor(session_id, actor_id, created_at_ts, updated_at_ts, last_step)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (session_id, actor_id, now, now, step),
                )
            else:
                conn.execute(
                    "UPDATE session_actor SET actor_id = ?, updated_at_ts = ?, last_step = ? WHERE session_id = ?",
                    (actor_id, now, step, session_id),
                )

            state_row = conn.execute(
                "SELECT scars_json, last_step, expires_at_step FROM actor_state WHERE actor_id = ?",
                (actor_id,),
            ).fetchone()

            before = np.zeros(self.k, dtype=float)
            after = np.zeros(self.k, dtype=float)
            if state_row is not None:
                before = _to_vec(_json_list(state_row["scars_json"], [0.0] * self.k), self.k)
                last_step = int(state_row["last_step"])
                expires_at = int(state_row["expires_at_step"])
                if step <= expires_at:
                    after = self._decay_scars(before, step - last_step)

            conn.execute(
                """
                INSERT INTO actor_state(actor_id, scars_json, last_step, expires_at_step, updated_at_ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(actor_id) DO UPDATE SET
                  scars_json = excluded.scars_json,
                  last_step = excluded.last_step,
                  expires_at_step = excluded.expires_at_step,
                  updated_at_ts = excluded.updated_at_ts
                """,
                (actor_id, json.dumps(after.tolist()), step, step + self.actor_ttl_steps, now),
            )

            freeze_until: Optional[int] = None
            freeze_active = False
            if self.transfer_freeze and self.freeze_scope == "actor":
                fr = conn.execute(
                    "SELECT freeze_until_step FROM actor_freeze WHERE actor_id = ?",
                    (actor_id,),
                ).fetchone()
                if fr is not None:
                    freeze_until = int(fr["freeze_until_step"])
                    freeze_active = freeze_until >= step

            q_count = 0
            if self.transfer_quarantine and self.quarantine_scope == "global_source":
                row = conn.execute(
                    "SELECT COUNT(1) AS n FROM source_quarantine WHERE quarantined_until_step >= ?",
                    (step,),
                ).fetchone()
                q_count = int(row["n"]) if row is not None else 0

            hydrated = HydratedState(
                actor_id=actor_id,
                actor_hash=self._actor_hash(actor_id),
                global_step=step,
                carryover_applied=bool(np.any(after > 0.0) or freeze_active or q_count > 0),
                carried_scars_before_decay=before,
                carried_scars_after_decay=after,
                freeze_until_step=freeze_until if freeze_active else None,
                quarantined_sources_count=q_count,
            )

        self._last_hydrated[(actor_id, session_id)] = hydrated
        return hydrated

    def record_step(
        self,
        actor_id: str,
        session_id: str,
        step_result: Any,
        policy_actions: List[OffAction],
        packet_items: List[Any],
    ) -> None:
        if not self.enabled:
            return
        del packet_items  # source ids are carried by SOURCE_QUARANTINE actions.

        now = _utc_now_iso()
        with self._connect() as conn:
            step = self._resolve_run_step(conn, actor_id=actor_id, session_id=session_id)
            scars = _to_vec(getattr(step_result, "m_next", np.zeros(self.k, dtype=float)), self.k)
            if not self.transfer_scars:
                scars = np.zeros(self.k, dtype=float)
            conn.execute(
                """
                INSERT INTO actor_state(actor_id, scars_json, last_step, expires_at_step, updated_at_ts)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(actor_id) DO UPDATE SET
                  scars_json = excluded.scars_json,
                  last_step = excluded.last_step,
                  expires_at_step = excluded.expires_at_step,
                  updated_at_ts = excluded.updated_at_ts
                """,
                (actor_id, json.dumps(scars.tolist()), step, step + self.actor_ttl_steps, now),
            )

            for action in policy_actions:
                if action.type == "TOOL_FREEZE" and self.transfer_freeze and self.freeze_scope == "actor":
                    horizon = max(0, int(action.horizon_steps or 0))
                    until_step = step + horizon
                    prev = conn.execute(
                        "SELECT freeze_until_step FROM actor_freeze WHERE actor_id = ?",
                        (actor_id,),
                    ).fetchone()
                    prev_until = int(prev["freeze_until_step"]) if prev is not None else -1
                    if until_step < prev_until:
                        continue
                    conn.execute(
                        """
                        INSERT INTO actor_freeze(actor_id, freeze_until_step, tool_mode, allowlist_json, updated_at_ts)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(actor_id) DO UPDATE SET
                          freeze_until_step = excluded.freeze_until_step,
                          tool_mode = excluded.tool_mode,
                          allowlist_json = excluded.allowlist_json,
                          updated_at_ts = excluded.updated_at_ts
                        """,
                        (
                            actor_id,
                            until_step,
                            action.tool_mode or "TOOLS_DISABLED",
                            json.dumps(list(action.allowlist or [])),
                            now,
                        ),
                    )

                if action.type == "SOURCE_QUARANTINE" and self.transfer_quarantine:
                    source_ids = list(action.source_ids or [])
                    if not source_ids:
                        continue
                    horizon = max(0, int(action.horizon_steps or self.source_quarantine_horizon_steps))
                    until_step = step + horizon
                    for source_id in source_ids:
                        row = conn.execute(
                            "SELECT strikes, quarantined_until_step FROM source_quarantine WHERE source_id = ?",
                            (source_id,),
                        ).fetchone()
                        strikes = (int(row["strikes"]) + 1) if row is not None else 1
                        prev_until = int(row["quarantined_until_step"]) if row is not None else -1
                        next_until = prev_until
                        if strikes >= self.strikes_to_quarantine and until_step > prev_until:
                            next_until = until_step
                        conn.execute(
                            """
                            INSERT INTO source_quarantine(source_id, strikes, quarantined_until_step, updated_at_ts)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(source_id) DO UPDATE SET
                              strikes = excluded.strikes,
                              quarantined_until_step = excluded.quarantined_until_step,
                              updated_at_ts = excluded.updated_at_ts
                            """,
                            (source_id, strikes, next_until, now),
                        )

            self._prune_expired_conn(conn, step)

    def active_actions(self, actor_id: str, session_id: str, step: int) -> List[OffAction]:
        del step
        if not self.enabled:
            return []

        with self._connect() as conn:
            cur_step = self._resolve_run_step(conn, actor_id=actor_id, session_id=session_id)
            self._prune_expired_conn(conn, cur_step)
            actions: List[OffAction] = []

            if self.transfer_freeze and self.freeze_scope == "actor":
                fr = conn.execute(
                    "SELECT freeze_until_step, tool_mode, allowlist_json FROM actor_freeze WHERE actor_id = ?",
                    (actor_id,),
                ).fetchone()
                if fr is not None and int(fr["freeze_until_step"]) >= cur_step:
                    until_step = int(fr["freeze_until_step"])
                    actions.append(
                        OffAction(
                            type="TOOL_FREEZE",
                            target="TOOLS",
                            tool_mode=str(fr["tool_mode"]),
                            allowlist=[str(v) for v in _json_list(fr["allowlist_json"], [])],
                            horizon_steps=max(0, until_step - cur_step),
                        )
                    )

            if self.transfer_quarantine and self.quarantine_scope == "global_source":
                rows = conn.execute(
                    """
                    SELECT source_id, quarantined_until_step
                    FROM source_quarantine
                    WHERE quarantined_until_step >= ?
                    ORDER BY source_id ASC
                    """,
                    (cur_step,),
                ).fetchall()
                if rows:
                    source_ids = [str(row["source_id"]) for row in rows]
                    max_until = max(int(row["quarantined_until_step"]) for row in rows)
                    actions.append(
                        OffAction(
                            type="SOURCE_QUARANTINE",
                            target="SOURCE",
                            source_ids=source_ids,
                            horizon_steps=max(0, max_until - cur_step),
                        )
                    )

            return actions

    def snapshot(self, actor_id: str, session_id: str, step: int) -> Dict[str, Any]:
        del step
        if not self.enabled:
            return {
                "freeze": {"active": False, "mode": "TOOLS_DISABLED", "allowlist": [], "freeze_until_step": None, "remaining_horizon": 0},
                "quarantine": {"active": False, "quarantined_sources": [], "total_quarantined": 0},
                "cross_session": {
                    "actor_hash": self._actor_hash(actor_id),
                    "carryover_applied": False,
                    "carried_scars_before_decay": [0.0] * self.k,
                    "carried_scars_after_decay": [0.0] * self.k,
                    "freeze_until_step": None,
                    "quarantined_sources_count": 0,
                    "state_backend": "disabled",
                    "state_version": "1.0",
                },
            }

        with self._connect() as conn:
            cur_step = self._resolve_run_step(conn, actor_id=actor_id, session_id=session_id)
            self._prune_expired_conn(conn, cur_step)

            fr = conn.execute(
                "SELECT freeze_until_step, tool_mode, allowlist_json FROM actor_freeze WHERE actor_id = ?",
                (actor_id,),
            ).fetchone()
            freeze_active = bool(fr is not None and int(fr["freeze_until_step"]) >= cur_step)
            freeze_until = int(fr["freeze_until_step"]) if freeze_active else None
            freeze_mode = str(fr["tool_mode"]) if freeze_active else "TOOLS_DISABLED"
            freeze_allow = [str(v) for v in _json_list(fr["allowlist_json"], [])] if freeze_active else []

            q_rows = conn.execute(
                """
                SELECT source_id, quarantined_until_step
                FROM source_quarantine
                WHERE quarantined_until_step >= ?
                ORDER BY source_id ASC
                """,
                (cur_step,),
            ).fetchall()
            quarantined = [
                {
                    "source_id": str(row["source_id"]),
                    "until_step": int(row["quarantined_until_step"]),
                    "remaining_horizon": max(0, int(row["quarantined_until_step"]) - cur_step),
                }
                for row in q_rows
            ]

        hydrated = self._last_hydrated.get((actor_id, session_id))
        if hydrated is None:
            hydrated = HydratedState(
                actor_id=actor_id,
                actor_hash=self._actor_hash(actor_id),
                global_step=0,
                carryover_applied=False,
                carried_scars_before_decay=np.zeros(self.k, dtype=float),
                carried_scars_after_decay=np.zeros(self.k, dtype=float),
                freeze_until_step=None,
                quarantined_sources_count=0,
            )

        return {
            "freeze": {
                "active": freeze_active,
                "mode": freeze_mode,
                "allowlist": freeze_allow,
                "freeze_until_step": freeze_until,
                "remaining_horizon": max(0, (freeze_until or 0) - hydrated.global_step) if freeze_active else 0,
            },
            "quarantine": {
                "active": bool(quarantined),
                "quarantined_sources": quarantined,
                "total_quarantined": len(quarantined),
            },
            "cross_session": {
                "actor_hash": hydrated.actor_hash,
                "carryover_applied": bool(hydrated.carryover_applied),
                "carried_scars_before_decay": hydrated.carried_scars_before_decay.tolist(),
                "carried_scars_after_decay": hydrated.carried_scars_after_decay.tolist(),
                "freeze_until_step": freeze_until,
                "quarantined_sources_count": len(quarantined),
                "state_backend": "sqlite",
                "state_version": hydrated.state_version,
            },
        }

    def reset_actor(self, actor_id: str) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            conn.execute("DELETE FROM actor_state WHERE actor_id = ?", (actor_id,))
            conn.execute("DELETE FROM actor_freeze WHERE actor_id = ?", (actor_id,))
            conn.execute("DELETE FROM session_actor WHERE actor_id = ?", (actor_id,))

        to_delete = [key for key in self._last_hydrated.keys() if key[0] == actor_id]
        for key in to_delete:
            self._last_hydrated.pop(key, None)

    def prune_expired(self) -> None:
        if not self.enabled:
            return
        with self._connect() as conn:
            step = self._current_global_step(conn)
            self._prune_expired_conn(conn, step)

    def _prune_expired_conn(self, conn: sqlite3.Connection, cur_step: int) -> None:
        conn.execute("DELETE FROM actor_state WHERE expires_at_step < ?", (cur_step,))
        conn.execute("DELETE FROM actor_freeze WHERE freeze_until_step < ?", (cur_step,))
