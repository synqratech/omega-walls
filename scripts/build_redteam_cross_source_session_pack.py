from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_label(side: str) -> str:
    return "attack" if str(side).strip().lower() == "attack" else "benign"


def _safe_fragments(sample: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    frags = sample.get("source_fragments")
    if not isinstance(frags, list):
        return []
    out: List[Mapping[str, Any]] = []
    for frag in frags:
        if isinstance(frag, Mapping):
            out.append(frag)
    return out


def _group_fragments_by_session(fragments: Sequence[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for frag in sorted(
        fragments,
        key=lambda f: (
            str(f.get("session_id", "")),
            int(f.get("turn_id", 0) or 0),
            int(f.get("relative_time_min", 0) or 0),
        ),
    ):
        sid = str(frag.get("session_id", "session_a")).strip() or "session_a"
        grouped.setdefault(sid, []).append(frag)
    return grouped


def _trust_to_source_type(trust: str) -> str:
    return "trusted" if str(trust).strip().lower() == "trusted" else "external_untrusted"


@dataclass(frozen=True)
class BuildStats:
    files_total: int
    pairs_total: int
    sessions_total: int
    rows_total: int
    attack_sessions: int
    benign_sessions: int


def _iter_pairs(payload: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, Mapping):
                yield row


def build_rows_from_file(path: Path, *, source_tag: str) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    counts = {"pairs": 0, "sessions": 0, "attack_sessions": 0, "benign_sessions": 0}
    for pair in _iter_pairs(payload):
        attack = pair.get("attack")
        benign = pair.get("benign_twin")
        if not isinstance(attack, Mapping) or not isinstance(benign, Mapping):
            continue
        counts["pairs"] += 1
        pair_id = str(attack.get("id", f"pair_{counts['pairs']}")).strip() or f"pair_{counts['pairs']}"
        pair_uid = f"{source_tag}::{pair_id}"
        family = str(attack.get("family", "cross_source_cocktail")).strip() or "cross_source_cocktail"

        for side_name, sample in (("attack", attack), ("benign", benign)):
            label_session = _safe_label(side_name)
            actor_id = f"rt_actor::{pair_uid}::{side_name}"
            grouped = _group_fragments_by_session(_safe_fragments(sample))
            if not grouped:
                text = str(sample.get("text", "")).strip()
                if not text:
                    continue
                grouped = {"session_a": [{"text": text, "turn_id": 1, "source_id": f"{pair_id}:{side_name}:text", "source_type": "other", "trust": "untrusted"}]}

            if label_session == "attack":
                counts["attack_sessions"] += len(grouped)
            else:
                counts["benign_sessions"] += len(grouped)

            for local_session, frags in sorted(grouped.items(), key=lambda kv: kv[0]):
                counts["sessions"] += 1
                session_id = f"rt::{pair_uid}::{side_name}::{local_session}"
                for idx, frag in enumerate(frags, start=1):
                    text = str(frag.get("text", "")).strip()
                    if not text:
                        continue
                    trust = str(frag.get("trust", "untrusted")).strip() or "untrusted"
                    rows.append(
                        {
                            "session_id": session_id,
                            "turn_id": int(idx),
                            "text": text,
                            "label_turn": label_session,
                            "label_session": label_session,
                            "family": family,
                            "source_ref": str(frag.get("source_id", f"{pair_uid}:{session_id}:{idx}")).strip()
                            or f"{pair_uid}:{session_id}:{idx}",
                            "source_type": _trust_to_source_type(trust),
                            "actor_id": actor_id,
                            "bucket": "cross_session",
                            "eval_slice": "context_required",
                            "meta_phase": str(frag.get("phase", "")).strip(),
                            "meta_rel_time_min": int(frag.get("relative_time_min", 0) or 0),
                        }
                    )
    return rows, counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build session-pack JSONL from redteam cross-source cocktail pairs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files with attack/benign_twin pairs.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--meta-out",
        default=None,
        help="Optional meta JSON path (default: <out>.meta.json).",
    )
    args = parser.parse_args()

    out_path = (ROOT / str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = (ROOT / str(args.meta_out)).resolve() if args.meta_out else out_path.with_suffix(".meta.json")

    all_rows: List[Dict[str, Any]] = []
    total_pairs = 0
    total_sessions = 0
    total_attack_sessions = 0
    total_benign_sessions = 0

    resolved_inputs: List[str] = []
    for raw in args.inputs:
        in_path = (ROOT / str(raw)).resolve()
        if not in_path.exists():
            raise FileNotFoundError(f"input file not found: {in_path}")
        resolved_inputs.append(str(in_path))
        source_tag = in_path.stem
        rows, counts = build_rows_from_file(in_path, source_tag=source_tag)
        all_rows.extend(rows)
        total_pairs += int(counts.get("pairs", 0))
        total_sessions += int(counts.get("sessions", 0))
        total_attack_sessions += int(counts.get("attack_sessions", 0))
        total_benign_sessions += int(counts.get("benign_sessions", 0))

    # Deterministic output ordering.
    all_rows.sort(
        key=lambda r: (
            str(r.get("actor_id", "")),
            str(r.get("session_id", "")),
            int(r.get("turn_id", 0) or 0),
        )
    )

    with out_path.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = BuildStats(
        files_total=len(resolved_inputs),
        pairs_total=total_pairs,
        sessions_total=total_sessions,
        rows_total=len(all_rows),
        attack_sessions=total_attack_sessions,
        benign_sessions=total_benign_sessions,
    )
    meta = {
        "status": "ok",
        "run_id": f"redteam_cross_source_pack_{_utc_compact_now()}",
        "inputs": resolved_inputs,
        "output_jsonl": str(out_path),
        "counts": {
            "files_total": int(stats.files_total),
            "pairs_total": int(stats.pairs_total),
            "sessions_total": int(stats.sessions_total),
            "rows_total": int(stats.rows_total),
            "attack_sessions": int(stats.attack_sessions),
            "benign_sessions": int(stats.benign_sessions),
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
