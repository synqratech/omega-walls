from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


RUNTIME_ALLOWED_KEYS = {
    "session_id",
    "turn_id",
    "text",
    "source_type",
    "source_id",
}

RUNTIME_FORBIDDEN_KEYS = {
    "label_turn",
    "label_session",
    "family",
    "bucket",
    "eval_slice",
    "source_ref",
    "actor_id",
    "meta_phase",
    "meta_rel_time_min",
    "template_name",
    "template_id",
}

LABELS_REQUIRED_KEYS = {
    "session_id",
    "turn_id",
    "label_turn",
    "label_session",
    "family",
    "bucket",
    "eval_slice",
    "source_ref",
    "actor_id",
}

OPAQUE_SESSION_RE = re.compile(r"^s_\d{6}$")
OPAQUE_ACTOR_RE = re.compile(r"^a_\d{6}$")
OPAQUE_SOURCE_RE = re.compile(r"^src_\d{6}$")

ID_HINT_RE = re.compile(
    r"(attack|benign|family|template|cocktail|distributed|roleplay|exfil|atk|ben|promptshield|wainject|agentdojo)",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for ln in path.read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _default_labels_path(runtime_pack_path: Path) -> Path:
    # Canonical dual contract:
    # <pack_root>/runtime/session_pack.jsonl
    # <pack_root>/labels/session_pack_labels.jsonl
    if runtime_pack_path.parent.name == "runtime":
        return runtime_pack_path.parent.parent / "labels" / "session_pack_labels.jsonl"
    return runtime_pack_path.parent / "labels" / "session_pack_labels.jsonl"


def _coerce_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return (str(row.get("session_id", "")).strip(), _coerce_int(row.get("turn_id"), default=0))


def _has_id_hint(value: str) -> bool:
    return bool(ID_HINT_RE.search(str(value or "")))


def audit_runtime_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(rows, start=1):
        keys = set(str(k) for k in row.keys())
        forbidden_present = sorted(k for k in keys if k in RUNTIME_FORBIDDEN_KEYS)
        if forbidden_present:
            errors.append(f"row#{idx}: forbidden_keys={forbidden_present}")

        unknown = sorted(k for k in keys if k not in RUNTIME_ALLOWED_KEYS)
        if unknown:
            errors.append(f"row#{idx}: unknown_runtime_keys={unknown}")

        sid = str(row.get("session_id", "")).strip()
        src = str(row.get("source_id", "")).strip()
        if not OPAQUE_SESSION_RE.match(sid):
            errors.append(f"row#{idx}: non_opaque_session_id={sid!r}")
        if not OPAQUE_SOURCE_RE.match(src):
            errors.append(f"row#{idx}: non_opaque_source_id={src!r}")
        if _has_id_hint(sid):
            errors.append(f"row#{idx}: session_id_contains_hint={sid!r}")
        if _has_id_hint(src):
            errors.append(f"row#{idx}: source_id_contains_hint={src!r}")
    return errors


def audit_labels_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    errors: List[str] = []
    for idx, row in enumerate(rows, start=1):
        keys = set(str(k) for k in row.keys())
        missing = sorted(k for k in LABELS_REQUIRED_KEYS if k not in keys)
        if missing:
            errors.append(f"labels_row#{idx}: missing_required={missing}")
        sid = str(row.get("session_id", "")).strip()
        aid = str(row.get("actor_id", "")).strip()
        if not OPAQUE_SESSION_RE.match(sid):
            errors.append(f"labels_row#{idx}: non_opaque_session_id={sid!r}")
        if not OPAQUE_ACTOR_RE.match(aid):
            errors.append(f"labels_row#{idx}: non_opaque_actor_id={aid!r}")
        if _has_id_hint(sid):
            errors.append(f"labels_row#{idx}: session_id_contains_hint={sid!r}")
        if _has_id_hint(aid):
            errors.append(f"labels_row#{idx}: actor_id_contains_hint={aid!r}")
    return errors


def load_dual_session_pack(
    *,
    runtime_pack_path: Path,
    labels_pack_path: Optional[Path] = None,
    allow_legacy_runtime_leakage: bool = False,
) -> List[Dict[str, Any]]:
    runtime_rows = _load_jsonl(runtime_pack_path)
    if not runtime_rows:
        raise ValueError(f"no runtime rows loaded: {runtime_pack_path}")

    labels_path = labels_pack_path or _default_labels_path(runtime_pack_path)
    labels_rows = _load_jsonl(labels_path)

    # Legacy mode: combined rows in runtime payload (not dual-file contract).
    if not labels_rows:
        if not allow_legacy_runtime_leakage:
            raise ValueError(
                "labels sidecar not found; dual-file contract required. "
                f"expected labels file at: {labels_path}"
            )
        out: List[Dict[str, Any]] = []
        for row in runtime_rows:
            out.append(dict(row))
        return out

    runtime_errors = audit_runtime_rows(runtime_rows)
    labels_errors = audit_labels_rows(labels_rows)
    if (runtime_errors or labels_errors) and not allow_legacy_runtime_leakage:
        detail = runtime_errors + labels_errors
        detail_preview = "; ".join(detail[:12])
        raise ValueError(f"runtime payload leakage audit failed: {detail_preview}")

    labels_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for row in labels_rows:
        k = _key(row)
        labels_by_key[k] = dict(row)

    out_rows: List[Dict[str, Any]] = []
    for r in runtime_rows:
        k = _key(r)
        meta = labels_by_key.get(k)
        if meta is None:
            raise ValueError(f"missing labels sidecar row for key={k}")
        merged = dict(r)
        for mk, mv in meta.items():
            if mk in {"session_id", "turn_id"}:
                continue
            merged[mk] = mv
        out_rows.append(merged)
    return out_rows


def build_opaque_maps(
    *,
    rows: Sequence[Mapping[str, Any]],
    session_key: str = "session_id",
    actor_key: str = "actor_id",
    source_key: str = "source_ref",
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    session_values = sorted({str(r.get(session_key, "")).strip() for r in rows if str(r.get(session_key, "")).strip()})
    actor_values = sorted({str(r.get(actor_key, "")).strip() for r in rows if str(r.get(actor_key, "")).strip()})
    source_values = sorted({str(r.get(source_key, "")).strip() for r in rows if str(r.get(source_key, "")).strip()})

    s_map = {v: f"s_{idx:06d}" for idx, v in enumerate(session_values, start=1)}
    a_map = {v: f"a_{idx:06d}" for idx, v in enumerate(actor_values, start=1)}
    src_map = {v: f"src_{idx:06d}" for idx, v in enumerate(source_values, start=1)}
    return s_map, a_map, src_map

