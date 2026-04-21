from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import yaml

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_BASELINE_POINTERS = [
    "artifacts/wainject_eval/BASELINE_LATEST.json",
    "artifacts/session_eval/BASELINE_LATEST.json",
    "artifacts/promptshield_eval/BASELINE_LATEST.json",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_doc(path: Path, fmt: str) -> Any:
    raw = _read_text(path)
    if fmt == "json":
        return json.loads(raw)
    if fmt == "yaml":
        return yaml.safe_load(raw)
    raise ValueError(f"Unsupported format: {fmt}")


def _dump_doc(payload: Any, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=True, indent=2) + "\n"
    if fmt == "yaml":
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)
    raise ValueError(f"Unsupported format: {fmt}")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _get_dotted(payload: Mapping[str, Any], dotted: str) -> Any:
    cur: Any = payload
    for key in str(dotted).split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            raise KeyError(f"dotted path not found: {dotted}")
        cur = cur[key]
    return cur


def _set_dotted(payload: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = str(dotted).split(".")
    cur: Dict[str, Any] = payload
    for key in parts[:-1]:
        node = cur.get(key)
        if not isinstance(node, dict):
            node = {}
            cur[key] = node
        cur = node
    cur[parts[-1]] = value


def _load_flags_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("flags", [])
    if not isinstance(rows, list):
        raise ValueError("flags manifest must contain flags[]")
    out: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        file = str(row.get("file", "")).strip()
        fmt = str(row.get("format", "")).strip().lower()
        dotted = str(row.get("path", "")).strip()
        if not (name and file and fmt and dotted):
            continue
        if fmt not in {"yaml", "json"}:
            raise ValueError(f"unsupported format in flags manifest: {fmt}")
        out[name] = {"file": file, "format": fmt, "path": dotted}
    return out


def _write_op_log(payload: Dict[str, Any]) -> str:
    log_root = ROOT / "artifacts" / "ops_recovery"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    op_name = str(payload.get("operation", "op")).strip() or "op"
    path = log_root / f"{stamp}_{op_name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    history = log_root / "enterprise_rollback_history.jsonl"
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return str(path.relative_to(ROOT).as_posix())


def snapshot_state(
    *,
    baseline_pointers: List[str],
    flags_manifest: str,
    out: str,
) -> Dict[str, Any]:
    flags = _load_flags_manifest((ROOT / flags_manifest).resolve())
    pointer_rows: List[Dict[str, Any]] = []
    for rel in baseline_pointers:
        path = (ROOT / rel).resolve()
        row = {"path": rel, "exists": path.exists()}
        if path.exists():
            text = _read_text(path)
            row["sha256"] = _sha256_text(text)
            row["content"] = text
        pointer_rows.append(row)

    flag_rows: List[Dict[str, Any]] = []
    for name, spec in flags.items():
        file_path = (ROOT / spec["file"]).resolve()
        if not file_path.exists():
            flag_rows.append({"name": name, "error": "file_not_found", "file": spec["file"]})
            continue
        doc = _load_doc(file_path, spec["format"])
        value = _get_dotted(doc, spec["path"])
        flag_rows.append(
            {
                "name": name,
                "file": spec["file"],
                "format": spec["format"],
                "path": spec["path"],
                "value": value,
            }
        )

    payload = {
        "event": "enterprise_rollback_snapshot_v1",
        "timestamp": _utc_now(),
        "baseline_pointers": pointer_rows,
        "feature_flags": flag_rows,
        "flags_manifest": flags_manifest,
    }
    out_path = (ROOT / out).resolve()
    if out_path.suffix.lower() != ".json":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"snapshot_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    _ensure_parent(out_path)
    out_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
    payload["snapshot_file"] = str(out_path.relative_to(ROOT).as_posix())
    return payload


def restore_baseline(*, snapshot: str, dry_run: bool) -> Dict[str, Any]:
    snap_path = (ROOT / snapshot).resolve()
    payload = json.loads(snap_path.read_text(encoding="utf-8"))
    rows = payload.get("baseline_pointers", [])
    if not isinstance(rows, list):
        raise ValueError("invalid snapshot: baseline_pointers[] is required")

    changes: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = str(row.get("path", "")).strip()
        if not rel:
            continue
        path = (ROOT / rel).resolve()
        before = _read_text(path) if path.exists() else ""
        after = str(row.get("content", "")) if bool(row.get("exists", False)) else ""
        changed = before != after
        entry = {
            "path": rel,
            "changed": bool(changed),
            "before_sha256": _sha256_text(before) if before else None,
            "after_sha256": _sha256_text(after) if after else None,
        }
        if changed and not dry_run:
            _ensure_parent(path)
            if after:
                path.write_text(after, encoding="utf-8")
            elif path.exists():
                path.unlink()
        changes.append(entry)

    return {
        "event": "enterprise_rollback_restore_baseline_v1",
        "timestamp": _utc_now(),
        "dry_run": bool(dry_run),
        "snapshot": snapshot,
        "changes": changes,
        "changed_total": int(sum(1 for row in changes if bool(row.get("changed", False)))),
    }


def rollback_flag(*, snapshot: str, name: str, flags_manifest: str, dry_run: bool) -> Dict[str, Any]:
    spec_map = _load_flags_manifest((ROOT / flags_manifest).resolve())
    if name not in spec_map:
        raise ValueError(f"flag is not allowlisted: {name}")
    spec = spec_map[name]
    snap_payload = json.loads((ROOT / snapshot).resolve().read_text(encoding="utf-8"))
    snap_rows = snap_payload.get("feature_flags", [])
    if not isinstance(snap_rows, list):
        raise ValueError("invalid snapshot: feature_flags[] is required")
    snap_row = next((row for row in snap_rows if isinstance(row, dict) and str(row.get("name", "")) == name), None)
    if snap_row is None:
        raise ValueError(f"flag not found in snapshot: {name}")
    target_value = snap_row.get("value")

    file_path = (ROOT / spec["file"]).resolve()
    doc = _load_doc(file_path, spec["format"])
    before_value = _get_dotted(doc, spec["path"])
    changed = before_value != target_value
    if changed and not dry_run:
        if not isinstance(doc, dict):
            raise ValueError(f"document must be mapping for dotted path write: {spec['file']}")
        _set_dotted(doc, spec["path"], target_value)
        file_path.write_text(_dump_doc(doc, spec["format"]), encoding="utf-8")

    return {
        "event": "enterprise_rollback_flag_v1",
        "timestamp": _utc_now(),
        "dry_run": bool(dry_run),
        "snapshot": snapshot,
        "flag": name,
        "file": spec["file"],
        "path": spec["path"],
        "before": before_value,
        "after": target_value,
        "changed": bool(changed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Enterprise rollback controls with dry-run support.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snapshot = sub.add_parser("snapshot")
    p_snapshot.add_argument("--baseline-pointer", action="append", default=[])
    p_snapshot.add_argument("--flags-manifest", default="enterprise/config/rollback_feature_flags_allowlist.json")
    p_snapshot.add_argument("--out", default="artifacts/ops_recovery/rollback_snapshots")

    p_restore = sub.add_parser("restore-baseline")
    p_restore.add_argument("--snapshot", required=True)
    p_restore.add_argument("--dry-run", action="store_true")

    p_flag = sub.add_parser("rollback-flag")
    p_flag.add_argument("--snapshot", required=True)
    p_flag.add_argument("--name", required=True)
    p_flag.add_argument("--flags-manifest", default="enterprise/config/rollback_feature_flags_allowlist.json")
    p_flag.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    if args.cmd == "snapshot":
        pointers = list(args.baseline_pointer or []) or list(DEFAULT_BASELINE_POINTERS)
        result = snapshot_state(
            baseline_pointers=pointers,
            flags_manifest=str(args.flags_manifest),
            out=str(args.out),
        )
        op_payload = {"operation": "enterprise_rollback_snapshot", **result}
        op_payload["log_file"] = _write_op_log(op_payload)
        print(json.dumps(op_payload, ensure_ascii=True, indent=2))
        return 0
    if args.cmd == "restore-baseline":
        result = restore_baseline(snapshot=str(args.snapshot), dry_run=bool(args.dry_run))
        op_payload = {"operation": "enterprise_restore_baseline", **result}
        op_payload["log_file"] = _write_op_log(op_payload)
        print(json.dumps(op_payload, ensure_ascii=True, indent=2))
        return 0
    if args.cmd == "rollback-flag":
        result = rollback_flag(
            snapshot=str(args.snapshot),
            name=str(args.name),
            flags_manifest=str(args.flags_manifest),
            dry_run=bool(args.dry_run),
        )
        op_payload = {"operation": "enterprise_rollback_flag", **result}
        op_payload["log_file"] = _write_op_log(op_payload)
        print(json.dumps(op_payload, ensure_ascii=True, indent=2))
        return 0
    raise ValueError(f"unsupported command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
