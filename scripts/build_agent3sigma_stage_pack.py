from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "Agent3Sigma-Stage" / "data" / "advance"
DEFAULT_OUT_ROOT = ROOT / "tests" / "data" / "session_benchmark" / "agent3sigma_stage_advance_v1"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _snake_case(value: str) -> str:
    raw = str(value or "").strip().lower()
    out: List[str] = []
    prev_underscore = False
    for ch in raw:
        if ch.isalnum():
            out.append(ch)
            prev_underscore = False
        else:
            if not prev_underscore:
                out.append("_")
                prev_underscore = True
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "unknown"


def _load_json(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list: {path}")
    rows: List[Dict[str, Any]] = []
    for item in payload:
        if isinstance(item, Mapping):
            rows.append(dict(item))
    return rows


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compose_turn_text(turn: Mapping[str, Any]) -> str:
    user = _safe_text(turn.get("user"))
    tool_response = _safe_text(turn.get("tool_response"))
    if user and tool_response:
        if tool_response in user:
            return user
        return f"{user}\n\n{tool_response}"
    return user or tool_response


def _turn_is_attack(turn: Mapping[str, Any]) -> bool:
    return bool(turn.get("injection_in_user")) or bool(turn.get("injection_in_tool"))


def _turn_label(turn: Mapping[str, Any], *, session_label: str) -> str:
    if session_label == "attack" and _turn_is_attack(turn):
        return "attack"
    if session_label == "benign":
        return "benign"
    return "attack" if _turn_is_attack(turn) else "benign"


def _build_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    session_label: str,
    start_index: int,
    source_index: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
    runtime_rows: List[Dict[str, Any]] = []
    label_rows: List[Dict[str, Any]] = []
    session_index = start_index
    next_source_index = source_index
    for rec in records:
        session_index += 1
        session_id = f"s_{session_index:06d}"
        actor_id = f"a_{session_index:06d}"
        family = _snake_case(str(rec.get("risk_category", "unknown")))
        scenario = _snake_case(str(rec.get("scenario", "unknown")))
        record_id = _safe_text(rec.get("id")) or f"record_{session_index:06d}"
        turns = rec.get("turns", [])
        if not isinstance(turns, list) or not turns:
            continue
        for turn_index, turn in enumerate(turns, start=1):
            if not isinstance(turn, Mapping):
                continue
            text = _compose_turn_text(turn)
            if not text:
                continue
            next_source_index += 1
            source_id = f"src_{next_source_index:06d}"
            runtime_rows.append(
                {
                    "session_id": session_id,
                    "turn_id": int(turn_index),
                    "text": text,
                    "source_type": "external_untrusted",
                    "source_id": source_id,
                }
            )
            label_rows.append(
                {
                    "session_id": session_id,
                    "turn_id": int(turn_index),
                    "label_turn": _turn_label(turn, session_label=session_label),
                    "label_session": session_label,
                    "family": family,
                    "bucket": scenario,
                    "eval_slice": "text_intrinsic",
                    "source_ref": f"a3s:{record_id}:{turn_index}",
                    "actor_id": actor_id,
                }
            )
    return runtime_rows, label_rows, session_index, next_source_index


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _session_summary(runtime_rows: Sequence[Mapping[str, Any]], label_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    sessions = sorted({str(row.get("session_id", "")).strip() for row in label_rows if str(row.get("session_id", "")).strip()})
    attack_sessions = sorted({str(row.get("session_id", "")).strip() for row in label_rows if str(row.get("label_session", "")).strip() == "attack"})
    benign_sessions = sorted({str(row.get("session_id", "")).strip() for row in label_rows if str(row.get("label_session", "")).strip() == "benign"})
    family_counts: Dict[str, int] = {}
    for row in label_rows:
        fam = _safe_text(row.get("family"))
        if fam:
            family_counts[fam] = family_counts.get(fam, 0) + 1
    return {
        "sessions_total": int(len(sessions)),
        "attack_sessions": int(len(attack_sessions)),
        "benign_sessions": int(len(benign_sessions)),
        "rows_total": int(len(runtime_rows)),
        "label_rows_total": int(len(label_rows)),
        "by_family_turns": dict(sorted(family_counts.items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the Agent3Sigma Stage advanced benchmark pack.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    args = parser.parse_args(list(argv) if argv is not None else None)

    out_root = Path(args.out_root).resolve()
    seeds = _load_json(SOURCE_DIR / "seeds.json")
    injected = _load_json(SOURCE_DIR / "injected.json")

    runtime_seed, labels_seed, seed_idx, source_idx = _build_rows(
        seeds,
        session_label="benign",
        start_index=0,
        source_index=0,
    )
    runtime_inj, labels_inj, _, _ = _build_rows(
        injected,
        session_label="attack",
        start_index=seed_idx,
        source_index=source_idx,
    )
    runtime_rows = runtime_seed + runtime_inj
    label_rows = labels_seed + labels_inj

    runtime_path = out_root / "runtime" / "session_pack.jsonl"
    labels_path = out_root / "labels" / "session_pack_labels.jsonl"
    _write_jsonl(runtime_path, runtime_rows)
    _write_jsonl(labels_path, label_rows)

    meta = {
        "schema_version": "agent3sigma_stage_advance_pack_v1",
        "generated_at_utc": _utc_now_iso(),
        "source_dir": str(SOURCE_DIR.resolve()),
        "source_files": {
            "seeds": {
                "path": str((SOURCE_DIR / "seeds.json").resolve()),
                "sha256": _file_sha256(SOURCE_DIR / "seeds.json"),
                "records": int(len(seeds)),
            },
            "injected": {
                "path": str((SOURCE_DIR / "injected.json").resolve()),
                "sha256": _file_sha256(SOURCE_DIR / "injected.json"),
                "records": int(len(injected)),
            },
        },
        "runtime_pack_path": str(runtime_path.resolve()),
        "labels_pack_path": str(labels_path.resolve()),
        "summary": _session_summary(runtime_rows, label_rows),
    }
    (out_root / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_root / "README.md").write_text(
        "Generated Agent3Sigma Stage advanced session pack for Omega benchmark registry.\n",
        encoding="utf-8",
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
