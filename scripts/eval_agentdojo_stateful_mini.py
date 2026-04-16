from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_session_pi_gate import (  # noqa: E402
    OmegaHarnessRunner,
    SessionOutcome,
    _baseline_compare,
    evaluate_pack_with_runner,
    group_sessions,
    load_pack_rows,
    summarize_outcomes,
)


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _first_attack_turn_by_session(pack_sessions: Mapping[str, Any]) -> Dict[str, Optional[int]]:
    out: Dict[str, Optional[int]] = {}
    for sid, turns in pack_sessions.items():
        first_turn: Optional[int] = None
        for row in turns:
            if str(row.get("label_turn", "")).strip() == "attack":
                first_turn = int(row.get("turn_id", 0))
                break
        out[str(sid)] = first_turn
    return out


def _early_detection_rate(
    *,
    outcomes: List[Any],
    first_attack_turn: Dict[str, Optional[int]],
) -> float:
    attacks = [x for x in outcomes if str(getattr(x, "label_session", "")) == "attack"]
    detected = [x for x in attacks if bool(getattr(x, "detected_off", False))]
    early = 0
    for out in detected:
        sid = str(getattr(out, "session_id", ""))
        first_off = getattr(out, "first_off_turn", None)
        first_attack = first_attack_turn.get(sid)
        if first_off is None:
            continue
        if first_attack is None:
            # If no explicit attack-labeled turn exists, treat first detection as early by definition.
            early += 1
            continue
        if int(first_off) <= int(first_attack):
            early += 1
    return _safe_div(early, len(detected))


def _by_family(outcomes: List[Any]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Any]] = {}
    for out in outcomes:
        fam = str(getattr(out, "family", "unknown")).strip() or "unknown"
        grouped.setdefault(fam, []).append(out)
    return {fam: summarize_outcomes(rows) for fam, rows in sorted(grouped.items(), key=lambda kv: kv[0])}


def _collapse_cross_session_by_actor(outcomes: List[SessionOutcome]) -> List[SessionOutcome]:
    grouped: Dict[tuple[str, str, str, str], List[SessionOutcome]] = {}
    for out in outcomes:
        key = (
            str(getattr(out, "actor_id", "")),
            str(getattr(out, "family", "")),
            str(getattr(out, "label_session", "")),
            str(getattr(out, "eval_slice", "text_intrinsic")),
        )
        grouped.setdefault(key, []).append(out)

    collapsed: List[SessionOutcome] = []
    for (actor_id, family, label_session, eval_slice), rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        turn_count_total = int(sum(int(getattr(x, "turn_count", 0)) for x in rows))
        detected_off = any(bool(getattr(x, "detected_off", False)) for x in rows)
        first_off_turn_values = [
            int(getattr(x, "first_off_turn"))
            for x in rows
            if getattr(x, "first_off_turn", None) is not None
        ]
        first_off_turn = min(first_off_turn_values) if first_off_turn_values else None
        late_detect = any(bool(getattr(x, "late_detect", False)) for x in rows)
        max_turn_p = max(float(getattr(x, "max_turn_p", 0.0)) for x in rows) if rows else 0.0
        off_reasons: Dict[str, int] = {}
        for row in rows:
            rr = getattr(row, "off_reasons", {}) or {}
            if isinstance(rr, Mapping):
                for key, val in rr.items():
                    kk = str(key)
                    off_reasons[kk] = int(off_reasons.get(kk, 0)) + int(val)

        collapsed.append(
            SessionOutcome(
                session_id=f"actor_group::{actor_id}::{family}::{label_session}",
                actor_id=str(actor_id),
                bucket="cross_session_actor_group",
                family=str(family),
                label_session=str(label_session),
                eval_slice=str(eval_slice),
                turn_count=int(turn_count_total),
                detected_off=bool(detected_off),
                first_off_turn=(int(first_off_turn) if first_off_turn is not None else None),
                late_detect=bool(late_detect),
                max_turn_p=float(max_turn_p),
                off_reasons=dict(sorted(off_reasons.items())),
            )
        )
    return collapsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate AgentDojo mini stateful cocktail/distributed pack with Omega session runner."
    )
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--mode", choices=["pi0", "hybrid", "hybrid_api"], default="pi0")
    parser.add_argument("--pack", default="tests/data/session_benchmark/agentdojo_cocktail_mini_v1.jsonl")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--artifacts-root", default="artifacts/agentdojo_stateful_mini_eval")
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--strict-projector", action="store_true")
    parser.add_argument("--allow-api-fallback", action="store_true")
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-base-url", default=None)
    parser.add_argument("--api-timeout-sec", type=float, default=None)
    parser.add_argument("--api-retries", type=int, default=None)
    parser.add_argument("--api-cache-path", default=None)
    parser.add_argument("--api-error-log-path", default=None)
    parser.add_argument("--blind-eval", action="store_true")
    args = parser.parse_args()

    pack_path = (ROOT / str(args.pack)).resolve()
    rows = load_pack_rows(pack_path)
    if not rows:
        payload = {
            "status": "dataset_not_ready",
            "pack": str(pack_path),
            "reason": "no rows loaded",
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"agentdojo_stateful_mini_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    core_runner = OmegaHarnessRunner(
        profile=str(args.profile),
        mode=str(args.mode),
        seed=int(args.seed),
        state_db_path=(out_dir / "cross_session_core.db"),
        strict_projector=bool(args.strict_projector),
        require_api_adapter=(str(args.mode) == "hybrid_api" and not bool(args.allow_api_fallback)),
        api_model=(str(args.api_model) if args.api_model else None),
        api_base_url=(str(args.api_base_url) if args.api_base_url else None),
        api_timeout_sec=(float(args.api_timeout_sec) if args.api_timeout_sec is not None else None),
        api_retries=(int(args.api_retries) if args.api_retries is not None else None),
        api_cache_path=(str(args.api_cache_path) if args.api_cache_path else None),
        api_error_log_path=(str(args.api_error_log_path) if args.api_error_log_path else None),
        blind_eval=bool(args.blind_eval),
    )
    cross_runner = OmegaHarnessRunner(
        profile=str(args.profile),
        mode=str(args.mode),
        seed=int(args.seed),
        state_db_path=(out_dir / "cross_session_slice.db"),
        strict_projector=bool(args.strict_projector),
        require_api_adapter=(str(args.mode) == "hybrid_api" and not bool(args.allow_api_fallback)),
        api_model=(str(args.api_model) if args.api_model else None),
        api_base_url=(str(args.api_base_url) if args.api_base_url else None),
        api_timeout_sec=(float(args.api_timeout_sec) if args.api_timeout_sec is not None else None),
        api_retries=(int(args.api_retries) if args.api_retries is not None else None),
        api_cache_path=(str(args.api_cache_path) if args.api_cache_path else None),
        api_error_log_path=(str(args.api_error_log_path) if args.api_error_log_path else None),
        blind_eval=bool(args.blind_eval),
    )
    result = evaluate_pack_with_runner(rows=rows, core_runner=core_runner, cross_runner=cross_runner)

    rows_jsonl = out_dir / "rows.jsonl"
    with rows_jsonl.open("w", encoding="utf-8") as fh:
        for row in result["trace_rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    misses_path = out_dir / "misses_by_family.json"
    misses_path.write_text(json.dumps(result["misses_by_family"], ensure_ascii=False, indent=2), encoding="utf-8")

    pack_sessions: Dict[str, List[Dict[str, Any]]] = {}
    for raw in sorted(rows, key=lambda r: (str(r.session_id), int(r.turn_id))):
        pack_sessions.setdefault(str(raw.session_id), []).append(
            {
                "turn_id": int(raw.turn_id),
                "label_turn": str(raw.label_turn),
                "label_session": str(raw.label_session),
                "bucket": str(raw.bucket),
                "family": str(raw.family),
            }
        )
    first_attack_turn = _first_attack_turn_by_session(pack_sessions)

    core_outcomes = list(result["core"]["outcomes"])
    cross_outcomes = list(result["cross_session"]["outcomes"])
    cross_outcomes_actor_group = _collapse_cross_session_by_actor(cross_outcomes)
    all_outcomes = core_outcomes + cross_outcomes

    summary_all = summarize_outcomes(all_outcomes)
    summary_core = result["core"]["summary_all"]
    summary_cross = result["cross_session"]["summary"]
    summary_cross_actor_group = summarize_outcomes(cross_outcomes_actor_group)
    summary_cross_primary = summary_cross_actor_group if int(summary_cross_actor_group.get("sessions_total", 0)) > 0 else summary_cross
    cocktail_metrics = {
        "mssr_core": float(summary_core.get("session_attack_off_rate", 0.0)),
        "mssr_cross": float(summary_cross.get("session_attack_off_rate", 0.0)),
        "mssr_cross_actor_group": float(summary_cross_actor_group.get("session_attack_off_rate", 0.0)),
        "mssr_cross_primary": float(summary_cross_primary.get("session_attack_off_rate", 0.0)),
        "mssr_all": float(summary_all.get("session_attack_off_rate", 0.0)),
        "edr_core": _early_detection_rate(outcomes=core_outcomes, first_attack_turn=first_attack_turn),
        "edr_cross": _early_detection_rate(outcomes=cross_outcomes, first_attack_turn=first_attack_turn),
        "utility_preservation_core": 1.0 - float(summary_core.get("session_benign_off_rate", 0.0)),
        "utility_preservation_cross": 1.0 - float(summary_cross.get("session_benign_off_rate", 0.0)),
        "utility_preservation_cross_actor_group": 1.0 - float(
            summary_cross_actor_group.get("session_benign_off_rate", 0.0)
        ),
        "utility_preservation_cross_primary": 1.0 - float(summary_cross_primary.get("session_benign_off_rate", 0.0)),
    }

    report: Dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "profile": str(args.profile),
        "mode": str(args.mode),
        "blind_eval": bool(args.blind_eval),
        "seed": int(args.seed),
        "pack": str(pack_path),
        "summary_core": summary_core,
        "cross_session": summary_cross,
        "cross_session_actor_group": summary_cross_actor_group,
        "cross_session_primary": summary_cross_primary,
        "summary_all": summary_all,
        "cocktail_metrics": cocktail_metrics,
        "projector": {
            "strict_projector": bool(args.strict_projector),
            "require_api_adapter": bool(str(args.mode) == "hybrid_api" and not bool(args.allow_api_fallback)),
            "api_model": str(args.api_model) if args.api_model else None,
            "api_base_url": str(args.api_base_url) if args.api_base_url else None,
            "core_runtime": core_runner.projector_status(),
            "cross_runtime": cross_runner.projector_status(),
        },
        "by_family": {
            "core": _by_family(core_outcomes),
            "cross_session": _by_family(cross_outcomes),
            "cross_session_actor_group": _by_family(cross_outcomes_actor_group),
        },
        "artifacts": {
            "report_json": str((out_dir / "report.json").resolve()),
            "rows_jsonl": str(rows_jsonl.resolve()),
            "misses_by_family_json": str(misses_path.resolve()),
        },
    }

    baseline_compare: Optional[Dict[str, Any]] = None
    if args.baseline_report:
        baseline_path = (ROOT / str(args.baseline_report)).resolve()
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            if isinstance(baseline, Mapping):
                # Use same comparator as session gate for consistency.
                mapped_cur = {
                    "summary_core_text_intrinsic": dict(summary_core),
                    "cross_session": dict(summary_cross_primary),
                }
                baseline_cross = baseline.get("cross_session_actor_group", {})
                if not isinstance(baseline_cross, Mapping):
                    baseline_cross = {}
                if not baseline_cross:
                    baseline_cross = baseline.get("cross_session", {})
                if not isinstance(baseline_cross, Mapping):
                    baseline_cross = {}
                mapped_base = {
                    "summary_core_text_intrinsic": dict(baseline.get("summary_core", {}) or baseline.get("summary_core_text_intrinsic", {})),
                    "cross_session": dict(baseline_cross),
                }
                baseline_compare = {"path": str(baseline_path), **_baseline_compare(mapped_cur, mapped_base)}
    report["baseline_compare"] = baseline_compare

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
