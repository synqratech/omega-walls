from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parent.parent
ATTACK_ONSET_PHASES = {
    "triggered_user_request",
    "reasoning_to_tool_escalation",
    "tool_plan",
    "verification_note",
}


@dataclass(frozen=True)
class BuiltPack:
    pack_root: Path
    runtime_pack_path: Path
    manifest_path: Path
    readme_path: Path
    stats: Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iter_pairs(payload: Any) -> Iterable[Mapping[str, Any]]:
    if not isinstance(payload, list):
        return
    for row in payload:
        if isinstance(row, Mapping):
            attack = row.get("attack")
            benign = row.get("benign_twin")
            if isinstance(attack, Mapping) and isinstance(benign, Mapping):
                yield row


def _safe_text(value: Any) -> str:
    return str(value or "").strip()


def _safe_source_type(trust: str) -> str:
    return "internal_trusted" if str(trust).strip().lower() == "trusted" else "external_untrusted"


def _group_fragments(sample: Mapping[str, Any]) -> Dict[str, List[Mapping[str, Any]]]:
    frags = sample.get("source_fragments", [])
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    if isinstance(frags, list):
        for frag in frags:
            if not isinstance(frag, Mapping):
                continue
            sid = _safe_text(frag.get("session_id")) or "session_a"
            grouped[sid].append(frag)
    for sid in list(grouped.keys()):
        grouped[sid] = sorted(
            grouped[sid],
            key=lambda x: (
                int(x.get("turn_id", 0) or 0),
                int(x.get("relative_time_min", 0) or 0),
                _safe_text(x.get("phase")),
            ),
        )
    return dict(grouped)


def _label_turn_attack(phase: str, idx: int, total: int, phase_list: Sequence[str]) -> str:
    ph = str(phase).strip()
    if ph in ATTACK_ONSET_PHASES:
        return "attack"
    # Safety guard for delayed trigger: if no attack phases were produced, mark final turn as attack.
    if idx == total and not any(p in ATTACK_ONSET_PHASES for p in phase_list):
        return "attack"
    return "benign"


def _build_rows_from_pairs(
    *,
    pairs: Sequence[Mapping[str, Any]],
    source_tag: str,
    sample_limit: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    scenes: List[Dict[str, Any]] = []
    attacks = 0
    benign = 0
    turns_attack = 0
    turns_benign = 0

    for idx, pair in enumerate(list(pairs)[:sample_limit], start=1):
        attack = pair.get("attack", {})
        benign_twin = pair.get("benign_twin", {})
        pair_id = _safe_text(attack.get("id")) or f"pair_{idx:04d}"
        source_name = _safe_text(pair.get("__source_name")) or "src"
        pair_uid = f"{source_name}:{pair_id}"
        family = _safe_text(attack.get("family")) or "xsrc_stateful_only"

        for side_name, sample in (("attack", attack), ("benign", benign_twin)):
            grouped = _group_fragments(sample)
            if not grouped:
                text = _safe_text(sample.get("text"))
                if not text:
                    continue
                grouped = {"session_a": [{"text": text, "phase": "delivery", "turn_id": 1, "relative_time_min": 0}]}

            for local_session, frags in sorted(grouped.items(), key=lambda kv: kv[0]):
                session_id = f"{source_tag}::{pair_uid}::{side_name}::{local_session}"
                actor_id = f"{source_tag}_actor::{pair_uid}::{side_name}"
                phases = [_safe_text(x.get("phase")) for x in frags]
                for turn_idx, frag in enumerate(frags, start=1):
                    text = _safe_text(frag.get("text"))
                    if not text:
                        continue
                    phase = _safe_text(frag.get("phase"))
                    if side_name == "attack":
                        label_turn = _label_turn_attack(phase=phase, idx=turn_idx, total=len(frags), phase_list=phases)
                        label_session = "attack"
                    else:
                        label_turn = "benign"
                        label_session = "benign"
                    rows.append(
                        {
                            "session_id": session_id,
                            "turn_id": int(turn_idx),
                            "text": text,
                            "label_turn": label_turn,
                            "label_session": label_session,
                            "family": family,
                            "source_ref": _safe_text(frag.get("source_id")) or f"{pair_uid}:{side_name}:{local_session}:{turn_idx}",
                            "source_type": _safe_source_type(_safe_text(frag.get("trust"))),
                            "actor_id": actor_id,
                            "bucket": "cross_session",
                            "eval_slice": "context_required",
                            "meta_phase": phase,
                            "meta_rel_time_min": int(frag.get("relative_time_min", 0) or 0),
                        }
                    )
                    if label_turn == "attack":
                        turns_attack += 1
                    else:
                        turns_benign += 1

                if side_name == "attack":
                    attacks += 1
                else:
                    benign += 1
                scenes.append(
                    {
                        "scene_id": f"stateful_only_scene_{len(scenes) + 1:04d}",
                        "session_id": session_id,
                        "label_session": ("attack" if side_name == "attack" else "benign"),
                        "pair_id": pair_uid,
                        "family": family,
                        "source_session": local_session,
                    }
                )

    rows_sorted = sorted(rows, key=lambda x: (str(x["session_id"]), int(x["turn_id"])))
    stats = {
        "sessions_total": int(attacks + benign),
        "attack_sessions": int(attacks),
        "benign_sessions": int(benign),
        "turns_total": int(len(rows_sorted)),
        "attack_turns": int(turns_attack),
        "benign_turns": int(turns_benign),
    }
    return rows_sorted, scenes, stats


def _write_pack(
    *,
    out_root: Path,
    pack_id: str,
    rows: Sequence[Mapping[str, Any]],
    scenes: Sequence[Mapping[str, Any]],
    stats: Mapping[str, Any],
    source_inputs: Sequence[Path],
) -> BuiltPack:
    pack_root = (out_root / pack_id).resolve()
    runtime_dir = pack_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_pack_path = runtime_dir / "session_pack.jsonl"
    with runtime_pack_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")

    manifest = list(scenes)
    manifest_path = pack_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    readme = [
        f"# {pack_id}",
        "",
        "Stateful-only support eval pack (delayed trigger, distributed context, hard benign near-miss).",
        "",
        f"- generated_at_utc: {_utc_now_iso()}",
        f"- source_inputs: {[str(p.resolve()) for p in source_inputs]}",
        f"- sessions_total: {int(stats.get('sessions_total', 0))}",
        f"- attack_sessions: {int(stats.get('attack_sessions', 0))}",
        f"- benign_sessions: {int(stats.get('benign_sessions', 0))}",
        f"- turns_total: {int(stats.get('turns_total', 0))}",
        f"- attack_turns: {int(stats.get('attack_turns', 0))}",
        f"- benign_turns: {int(stats.get('benign_turns', 0))}",
        "",
        "Labeling policy:",
        "- `label_session=attack` for attack side; `label_session=benign` for benign twin.",
        "- Attack turns are delayed: only trigger/reasoning/tool-plan/final-verification phases are `label_turn=attack`.",
        "- Early context/memory turns remain `label_turn=benign` to stress stateful detection.",
    ]
    readme_path = pack_root / "README.md"
    readme_path.write_text("\n".join(readme) + "\n", encoding="utf-8")

    meta = {
        "generated_at_utc": _utc_now_iso(),
        "pack_id": pack_id,
        "pack_root": str(pack_root),
        "runtime_pack_path": str(runtime_pack_path),
        "manifest_path": str(manifest_path),
        "readme_path": str(readme_path),
        "stats": dict(stats),
    }
    (pack_root / "pack_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return BuiltPack(
        pack_root=pack_root,
        runtime_pack_path=runtime_pack_path,
        manifest_path=manifest_path,
        readme_path=readme_path,
        stats=dict(stats),
    )


def _load_pairs(paths: Sequence[Path]) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_name = path.stem
        for pair in _iter_pairs(payload):
            row = dict(pair)
            row["__source_name"] = source_name
            out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a stateful-only support eval pack with delayed attack onset and hard benign near-miss."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "redteam/archive/cross_source_cocktail_v3_template_runA_n12.json",
            "redteam/archive/cross_source_cocktail_v3_template_runB_n12.json",
        ],
    )
    parser.add_argument("--out-root", default="data/eval_pack")
    parser.add_argument("--pack-id", default="omega_stateful_only_support_pack_v1")
    parser.add_argument("--source-tag", default="stateful_only")
    parser.add_argument("--sample-limit", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    input_paths = [(ROOT / str(p)).resolve() for p in list(args.inputs)]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"input not found: {path}")
    pairs = _load_pairs(input_paths)
    rng.shuffle(pairs)
    rows, scenes, stats = _build_rows_from_pairs(
        pairs=pairs,
        source_tag=str(args.source_tag),
        sample_limit=max(1, int(args.sample_limit)),
    )
    built = _write_pack(
        out_root=(ROOT / str(args.out_root)).resolve(),
        pack_id=str(args.pack_id),
        rows=rows,
        scenes=scenes,
        stats=stats,
        source_inputs=input_paths,
    )
    out = {
        "status": "ok",
        "pack_id": str(args.pack_id),
        "pack_root": str(built.pack_root),
        "runtime_pack_path": str(built.runtime_pack_path),
        "manifest_path": str(built.manifest_path),
        "readme_path": str(built.readme_path),
        "stats": built.stats,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
