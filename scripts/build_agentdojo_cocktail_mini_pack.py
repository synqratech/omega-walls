from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _norm_text(value: Any) -> str:
    text = str(value or "").strip()
    return " ".join(text.split())


def _short(text: str, max_chars: int) -> str:
    norm = _norm_text(text)
    if max_chars <= 0:
        return norm
    if len(norm) <= max_chars:
        return norm
    return norm[: max_chars - 3].rstrip() + "..."


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _message_texts(messages: Sequence[Dict[str, Any]], *, role: str) -> List[str]:
    out: List[str] = []
    for row in messages:
        if not isinstance(row, dict):
            continue
        if str(row.get("role", "")).strip().lower() != role:
            continue
        txt = _norm_text(row.get("content", ""))
        if txt:
            out.append(txt)
    return out


def _first_nonempty(values: Sequence[str]) -> str:
    for value in values:
        norm = _norm_text(value)
        if norm:
            return norm
    return ""


def _injection_payload_text(attack_payload: Dict[str, Any]) -> str:
    injections = attack_payload.get("injections", {})
    if not isinstance(injections, dict):
        return ""
    parts: List[str] = []
    for _, value in sorted(injections.items(), key=lambda kv: str(kv[0])):
        txt = _norm_text(value)
        if txt:
            parts.append(txt)
    return _norm_text("\n".join(parts))


@dataclass(frozen=True)
class AttackCase:
    suite: str
    user_task_id: str
    injection_task_id: str
    attack_file: Path
    benign_file: Path


def _iter_attack_cases(
    *,
    runs_root: Path,
    pipeline: str,
    suites: Sequence[str],
    attack_type: str,
) -> Iterable[AttackCase]:
    model_root = runs_root / pipeline
    if not model_root.exists():
        return []
    out: List[AttackCase] = []
    for suite in suites:
        suite_root = model_root / suite
        if not suite_root.exists():
            continue
        for user_dir in sorted([p for p in suite_root.iterdir() if p.is_dir() and p.name.startswith("user_task_")], key=lambda p: p.name):
            attack_dir = user_dir / attack_type
            benign_file = user_dir / "none" / "none.json"
            if not attack_dir.exists() or not benign_file.exists():
                continue
            for attack_file in sorted(attack_dir.glob("injection_task_*.json"), key=lambda p: p.name):
                injection_task_id = attack_file.stem
                out.append(
                    AttackCase(
                        suite=suite,
                        user_task_id=user_dir.name,
                        injection_task_id=injection_task_id,
                        attack_file=attack_file,
                        benign_file=benign_file,
                    )
                )
    return out


def _build_attack_turns(
    *,
    benign_payload: Dict[str, Any],
    attack_payload: Dict[str, Any],
    max_turn_chars: int,
) -> List[Tuple[str, str]]:
    benign_msgs = benign_payload.get("messages", [])
    attack_msgs = attack_payload.get("messages", [])
    if not isinstance(benign_msgs, list) or not isinstance(attack_msgs, list):
        return []

    benign_user = _first_nonempty(_message_texts(benign_msgs, role="user"))
    benign_tool = _first_nonempty(_message_texts(benign_msgs, role="tool"))
    attack_tool = _first_nonempty(_message_texts(attack_msgs, role="tool"))
    injection_text = _injection_payload_text(attack_payload)

    turns: List[Tuple[str, str]] = []
    if benign_user:
        turns.append((_short(benign_user, max_turn_chars), "benign"))
    if benign_tool:
        turns.append((_short(benign_tool, max_turn_chars), "benign"))
    if attack_tool:
        turns.append((_short(attack_tool, max_turn_chars), "attack"))
    if injection_text:
        turns.append((_short(injection_text, max_turn_chars), "attack"))

    dedup: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for txt, label in turns:
        key = _norm_text(txt)
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append((txt, label))

    if len(dedup) < 4:
        fallback = _short(_first_nonempty(_message_texts(attack_msgs, role="assistant")), max_turn_chars)
        if fallback and fallback not in seen:
            dedup.append((fallback, "attack"))
    if len(dedup) < 3:
        return []
    if len(dedup) == 3:
        # ensure gradual pattern benign -> benign -> attack -> attack
        dedup.append((dedup[-1][0], "attack"))
    return dedup[:4]


def _build_benign_turns(*, benign_payload: Dict[str, Any], max_turn_chars: int) -> List[str]:
    msgs = benign_payload.get("messages", [])
    if not isinstance(msgs, list):
        return []
    candidates = [
        _first_nonempty(_message_texts(msgs, role="user")),
        _first_nonempty(_message_texts(msgs, role="tool")),
        _first_nonempty(_message_texts(msgs, role="assistant")),
    ]
    out: List[str] = []
    seen: set[str] = set()
    for txt in candidates:
        norm = _norm_text(txt)
        if not norm:
            continue
        shr = _short(norm, max_turn_chars)
        if shr in seen:
            continue
        seen.add(shr)
        out.append(shr)
    return out[:3]


def build_agentdojo_cocktail_mini_pack(
    *,
    runs_root: Path,
    pipeline: str,
    attack_type: str,
    suites: Sequence[str],
    seed: int,
    max_attack_sessions: int,
    max_cross_pairs: int,
    max_turn_chars: int,
    out_jsonl: Path,
    out_meta_json: Path,
) -> Dict[str, Any]:
    rng = random.Random(int(seed))
    cases = list(_iter_attack_cases(runs_root=runs_root, pipeline=pipeline, suites=suites, attack_type=attack_type))
    if not cases:
        return {
            "status": "dataset_not_ready",
            "reason": "no attack cases found",
            "runs_root": str(runs_root),
            "pipeline": pipeline,
            "attack_type": attack_type,
        }

    rng.shuffle(cases)
    selected = cases[: max(1, int(max_attack_sessions))]

    rows: List[Dict[str, Any]] = []
    attack_sessions_created = 0
    benign_sessions_created = 0
    cross_attack_sessions_created = 0
    cross_benign_sessions_created = 0
    by_suite: Dict[str, int] = {}

    # Core attack + benign sessions
    for idx, case in enumerate(selected, start=1):
        benign_payload = _load_json(case.benign_file)
        attack_payload = _load_json(case.attack_file)
        attack_turns = _build_attack_turns(
            benign_payload=benign_payload,
            attack_payload=attack_payload,
            max_turn_chars=int(max_turn_chars),
        )
        benign_turns = _build_benign_turns(benign_payload=benign_payload, max_turn_chars=int(max_turn_chars))
        if len(attack_turns) < 4 or len(benign_turns) < 2:
            continue

        attack_session_id = f"agentdojo_core_atk_{idx:03d}"
        attack_actor_id = f"agentdojo_core_actor_{idx:03d}"
        for turn_id, (text, turn_label) in enumerate(attack_turns, start=1):
            rows.append(
                {
                    "session_id": attack_session_id,
                    "turn_id": int(turn_id),
                    "text": str(text),
                    "label_turn": str(turn_label),
                    "label_session": "attack",
                    "family": "cocktail",
                    "source_ref": str(case.attack_file.relative_to(ROOT).as_posix()) if case.attack_file.is_relative_to(ROOT) else str(case.attack_file.as_posix()),
                    "actor_id": attack_actor_id,
                    "bucket": "core",
                    "eval_slice": "text_intrinsic",
                }
            )
        attack_sessions_created += 1
        by_suite[case.suite] = by_suite.get(case.suite, 0) + 1

        benign_session_id = f"agentdojo_core_ben_{idx:03d}"
        benign_actor_id = f"agentdojo_core_ben_actor_{idx:03d}"
        for turn_id, text in enumerate(benign_turns, start=1):
            rows.append(
                {
                    "session_id": benign_session_id,
                    "turn_id": int(turn_id),
                    "text": str(text),
                    "label_turn": "benign",
                    "label_session": "benign",
                    "family": "benign_long_context",
                    "source_ref": str(case.benign_file.relative_to(ROOT).as_posix()) if case.benign_file.is_relative_to(ROOT) else str(case.benign_file.as_posix()),
                    "actor_id": benign_actor_id,
                    "bucket": "core",
                    "eval_slice": "text_intrinsic",
                }
            )
        benign_sessions_created += 1

    # Cross-session distributed split: benign prelude in session A, attack payload in session B
    for idx, case in enumerate(selected[: max(0, int(max_cross_pairs))], start=1):
        benign_payload = _load_json(case.benign_file)
        attack_payload = _load_json(case.attack_file)
        attack_turns = _build_attack_turns(
            benign_payload=benign_payload,
            attack_payload=attack_payload,
            max_turn_chars=int(max_turn_chars),
        )
        benign_turns = _build_benign_turns(benign_payload=benign_payload, max_turn_chars=int(max_turn_chars))
        if len(attack_turns) < 4 or len(benign_turns) < 2:
            continue

        actor_id = f"agentdojo_xs_actor_{idx:03d}"
        sess_a = f"agentdojo_xs_atk_{idx:03d}_a"
        sess_b = f"agentdojo_xs_atk_{idx:03d}_b"

        for turn_id, (text, _) in enumerate(attack_turns[:2], start=1):
            rows.append(
                {
                    "session_id": sess_a,
                    "turn_id": int(turn_id),
                    "text": str(text),
                    "label_turn": "benign",
                    "label_session": "attack",
                    "family": "distributed_wo_explicit",
                    "source_ref": str(case.attack_file.relative_to(ROOT).as_posix()) if case.attack_file.is_relative_to(ROOT) else str(case.attack_file.as_posix()),
                    "actor_id": actor_id,
                    "bucket": "cross_session",
                    "eval_slice": "text_intrinsic",
                }
            )
        for turn_id, (text, _) in enumerate(attack_turns[2:], start=1):
            rows.append(
                {
                    "session_id": sess_b,
                    "turn_id": int(turn_id),
                    "text": str(text),
                    "label_turn": "attack",
                    "label_session": "attack",
                    "family": "distributed_wo_explicit",
                    "source_ref": str(case.attack_file.relative_to(ROOT).as_posix()) if case.attack_file.is_relative_to(ROOT) else str(case.attack_file.as_posix()),
                    "actor_id": actor_id,
                    "bucket": "cross_session",
                    "eval_slice": "text_intrinsic",
                }
            )
        cross_attack_sessions_created += 2

        ben_actor_id = f"agentdojo_xs_ben_actor_{idx:03d}"
        ben_sess_a = f"agentdojo_xs_ben_{idx:03d}_a"
        ben_sess_b = f"agentdojo_xs_ben_{idx:03d}_b"
        for turn_id, text in enumerate(benign_turns[:1], start=1):
            rows.append(
                {
                    "session_id": ben_sess_a,
                    "turn_id": int(turn_id),
                    "text": str(text),
                    "label_turn": "benign",
                    "label_session": "benign",
                    "family": "benign_long_context",
                    "source_ref": str(case.benign_file.relative_to(ROOT).as_posix()) if case.benign_file.is_relative_to(ROOT) else str(case.benign_file.as_posix()),
                    "actor_id": ben_actor_id,
                    "bucket": "cross_session",
                    "eval_slice": "text_intrinsic",
                }
            )
        for turn_id, text in enumerate(benign_turns[1:], start=1):
            rows.append(
                {
                    "session_id": ben_sess_b,
                    "turn_id": int(turn_id),
                    "text": str(text),
                    "label_turn": "benign",
                    "label_session": "benign",
                    "family": "benign_long_context",
                    "source_ref": str(case.benign_file.relative_to(ROOT).as_posix()) if case.benign_file.is_relative_to(ROOT) else str(case.benign_file.as_posix()),
                    "actor_id": ben_actor_id,
                    "bucket": "cross_session",
                    "eval_slice": "text_intrinsic",
                }
            )
        cross_benign_sessions_created += 2

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            str(r["bucket"]),
            str(r["session_id"]),
            int(r["turn_id"]),
        ),
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for row in rows_sorted:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    session_ids = sorted({str(r["session_id"]) for r in rows_sorted})
    attack_session_ids = sorted({str(r["session_id"]) for r in rows_sorted if str(r["label_session"]) == "attack"})
    benign_session_ids = sorted({str(r["session_id"]) for r in rows_sorted if str(r["label_session"]) == "benign"})
    meta = {
        "status": "ok",
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": {
            "runs_root": str(runs_root),
            "pipeline": str(pipeline),
            "attack_type": str(attack_type),
            "suites": list(suites),
            "seed": int(seed),
            "max_attack_sessions": int(max_attack_sessions),
            "max_cross_pairs": int(max_cross_pairs),
            "max_turn_chars": int(max_turn_chars),
        },
        "counts": {
            "rows_total": int(len(rows_sorted)),
            "sessions_total": int(len(session_ids)),
            "attack_sessions_total": int(len(attack_session_ids)),
            "benign_sessions_total": int(len(benign_session_ids)),
            "core_attack_sessions": int(attack_sessions_created),
            "core_benign_sessions": int(benign_sessions_created),
            "cross_attack_sessions": int(cross_attack_sessions_created),
            "cross_benign_sessions": int(cross_benign_sessions_created),
            "selected_attack_cases": int(len(selected)),
        },
        "by_suite": dict(sorted(by_suite.items())),
        "artifacts": {
            "pack_jsonl": str(out_jsonl.resolve()),
            "meta_json": str(out_meta_json.resolve()),
        },
    }
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build small AgentDojo stateful cocktail/distributed mini-pack from cached runs."
    )
    parser.add_argument("--runs-root", default="data/AgentDojo/runs")
    parser.add_argument("--pipeline", default="gpt-4o-2024-05-13")
    parser.add_argument("--attack-type", default="important_instructions")
    parser.add_argument("--suites", default="banking,travel,slack,workspace")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-attack-sessions", type=int, default=12)
    parser.add_argument("--max-cross-pairs", type=int, default=4)
    parser.add_argument("--max-turn-chars", type=int, default=1500)
    parser.add_argument("--out", default="tests/data/session_benchmark/agentdojo_cocktail_mini_v1.jsonl")
    parser.add_argument("--meta-out", default="tests/data/session_benchmark/agentdojo_cocktail_mini_v1.meta.json")
    args = parser.parse_args()

    suites = [s.strip() for s in str(args.suites).split(",") if s.strip()]
    if not suites:
        raise ValueError("at least one suite must be provided via --suites")

    meta = build_agentdojo_cocktail_mini_pack(
        runs_root=(ROOT / str(args.runs_root)).resolve(),
        pipeline=str(args.pipeline),
        attack_type=str(args.attack_type),
        suites=suites,
        seed=int(args.seed),
        max_attack_sessions=int(args.max_attack_sessions),
        max_cross_pairs=int(args.max_cross_pairs),
        max_turn_chars=int(args.max_turn_chars),
        out_jsonl=(ROOT / str(args.out)).resolve(),
        out_meta_json=(ROOT / str(args.meta_out)).resolve(),
    )
    run_id = f"agentdojo_mini_pack_{_utc_compact_now()}"
    payload = {"run_id": run_id, **meta}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if str(meta.get("status")) == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
