from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.policy.cross_session_state import CrossSessionStateManager


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _connect(sqlite_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(sqlite_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _table_count(conn: sqlite3.Connection, table: str) -> int:
    row = conn.execute(f"SELECT COUNT(1) AS n FROM {table}").fetchone()
    return int(row["n"]) if row is not None else 0


def collect_state_snapshot(sqlite_path: Path, actor_id: str | None = None, max_rows: int = 20) -> Dict[str, Any]:
    with _connect(sqlite_path) as conn:
        global_step_row = conn.execute("SELECT value FROM meta WHERE key = 'global_step'").fetchone()
        global_step = int(global_step_row["value"]) if global_step_row is not None else 0

        counts = {
            "actor_state": _table_count(conn, "actor_state"),
            "actor_freeze": _table_count(conn, "actor_freeze"),
            "source_quarantine": _table_count(conn, "source_quarantine"),
            "session_actor": _table_count(conn, "session_actor"),
        }

        if actor_id:
            actor_state = conn.execute(
                "SELECT actor_id, scars_json, last_step, expires_at_step FROM actor_state WHERE actor_id = ?",
                (actor_id,),
            ).fetchall()
            actor_freeze = conn.execute(
                "SELECT actor_id, freeze_until_step, tool_mode, allowlist_json FROM actor_freeze WHERE actor_id = ?",
                (actor_id,),
            ).fetchall()
            session_actor = conn.execute(
                "SELECT session_id, actor_id, last_step FROM session_actor WHERE actor_id = ? ORDER BY session_id ASC",
                (actor_id,),
            ).fetchall()
        else:
            actor_state = conn.execute(
                "SELECT actor_id, scars_json, last_step, expires_at_step FROM actor_state ORDER BY actor_id ASC LIMIT ?",
                (max_rows,),
            ).fetchall()
            actor_freeze = conn.execute(
                "SELECT actor_id, freeze_until_step, tool_mode, allowlist_json FROM actor_freeze ORDER BY actor_id ASC LIMIT ?",
                (max_rows,),
            ).fetchall()
            session_actor = conn.execute(
                "SELECT session_id, actor_id, last_step FROM session_actor ORDER BY session_id ASC LIMIT ?",
                (max_rows,),
            ).fetchall()

        source_quarantine = conn.execute(
            """
            SELECT source_id, strikes, quarantined_until_step
            FROM source_quarantine
            ORDER BY quarantined_until_step DESC, source_id ASC
            LIMIT ?
            """,
            (max_rows,),
        ).fetchall()

    return {
        "global_step": global_step,
        "counts": counts,
        "actor_state": [dict(r) for r in actor_state],
        "actor_freeze": [dict(r) for r in actor_freeze],
        "source_quarantine": [dict(r) for r in source_quarantine],
        "session_actor": [dict(r) for r in session_actor],
    }


def clear_actor_freeze(sqlite_path: Path, actor_id: str) -> int:
    with _connect(sqlite_path) as conn:
        cur = conn.execute("DELETE FROM actor_freeze WHERE actor_id = ?", (actor_id,))
        conn.commit()
        return int(cur.rowcount or 0)


def clear_quarantine_source(sqlite_path: Path, source_id: str) -> int:
    with _connect(sqlite_path) as conn:
        cur = conn.execute("DELETE FROM source_quarantine WHERE source_id = ?", (source_id,))
        conn.commit()
        return int(cur.rowcount or 0)


def clear_quarantine_all(sqlite_path: Path, confirm: bool) -> int:
    if not confirm:
        raise ValueError("clear-quarantine-all requires --confirm")
    with _connect(sqlite_path) as conn:
        cur = conn.execute("DELETE FROM source_quarantine")
        conn.commit()
        return int(cur.rowcount or 0)


def _write_op_log(log_root: Path, payload: Dict[str, Any]) -> Path:
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    op_id = f"{stamp}_{payload['operation']}"
    event_path = log_root / f"{op_id}.json"
    event_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    history = log_root / "history.jsonl"
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return event_path


def _load_manager(profile: str) -> tuple[CrossSessionStateManager, Path]:
    snapshot = load_resolved_config(profile=profile)
    manager = CrossSessionStateManager.from_config(snapshot.resolved)
    if not manager.enabled:
        raise RuntimeError("cross_session manager is disabled in config")
    return manager, manager.sqlite_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Ops recovery controls for cross-session freeze/quarantine state")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--log-dir", default="artifacts/ops_recovery")

    sub = parser.add_subparsers(dest="command", required=True)

    p_snapshot = sub.add_parser("snapshot")
    p_snapshot.add_argument("--actor-id", default=None)
    p_snapshot.add_argument("--max-rows", type=int, default=20)

    sub.add_parser("prune-expired")

    p_reset = sub.add_parser("reset-actor")
    p_reset.add_argument("--actor-id", required=True)

    p_clear_freeze = sub.add_parser("clear-freeze")
    p_clear_freeze.add_argument("--actor-id", required=True)

    p_clear_qs = sub.add_parser("clear-quarantine-source")
    p_clear_qs.add_argument("--source-id", required=True)

    p_clear_all = sub.add_parser("clear-quarantine-all")
    p_clear_all.add_argument("--confirm", action="store_true")

    args = parser.parse_args()

    manager, sqlite_path = _load_manager(profile=args.profile)
    log_root = ROOT / args.log_dir

    before = collect_state_snapshot(sqlite_path=sqlite_path)
    result: Dict[str, Any] = {}

    if args.command == "snapshot":
        result = collect_state_snapshot(
            sqlite_path=sqlite_path,
            actor_id=args.actor_id,
            max_rows=max(1, int(args.max_rows)),
        )
    elif args.command == "prune-expired":
        manager.prune_expired()
        result = {"pruned": True}
    elif args.command == "reset-actor":
        manager.reset_actor(str(args.actor_id))
        result = {"actor_id": str(args.actor_id), "reset": True}
    elif args.command == "clear-freeze":
        removed = clear_actor_freeze(sqlite_path=sqlite_path, actor_id=str(args.actor_id))
        result = {"actor_id": str(args.actor_id), "removed_rows": removed}
    elif args.command == "clear-quarantine-source":
        removed = clear_quarantine_source(sqlite_path=sqlite_path, source_id=str(args.source_id))
        result = {"source_id": str(args.source_id), "removed_rows": removed}
    elif args.command == "clear-quarantine-all":
        removed = clear_quarantine_all(sqlite_path=sqlite_path, confirm=bool(args.confirm))
        result = {"removed_rows": removed}
    else:
        raise ValueError(f"Unsupported command: {args.command}")

    after = collect_state_snapshot(sqlite_path=sqlite_path)
    payload = {
        "event": "ops_state_recovery_v1",
        "timestamp": _now_utc_iso(),
        "operation": args.command,
        "profile": args.profile,
        "sqlite_path": str(sqlite_path.as_posix()),
        "before": before["counts"],
        "after": after["counts"],
        "result": result,
    }
    event_path = _write_op_log(log_root=log_root, payload=payload)
    payload["log_file"] = str(event_path.as_posix())
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
