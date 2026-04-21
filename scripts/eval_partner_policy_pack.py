from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OmegaState, WALLS_V1
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector


POLICY_PASS = "PASS"
POLICY_WARN = "WARN"
POLICY_BLOCK = "BLOCK"
POLICY_VALUES = {POLICY_PASS, POLICY_WARN, POLICY_BLOCK}
POLICY_RANK = {POLICY_PASS: 0, POLICY_WARN: 1, POLICY_BLOCK: 2}


@dataclass(frozen=True)
class PartnerStatelessCase:
    case_id: str
    family: str
    expected_policy: str
    expected_off: bool
    notes: str
    relative_path: str
    text: str


@dataclass(frozen=True)
class PartnerSessionTurn:
    case_id: str
    family: str
    session_id: str
    turn_id: int
    label_turn: str
    label_session: str
    expected_policy: str
    expected_off: bool
    notes: str
    relative_path: str
    source_ref: str
    actor_id: str
    bucket: str
    eval_slice: str
    text: str


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def map_control_outcome_to_partner_policy(control_outcome: str) -> str:
    v = str(control_outcome or "").strip().upper()
    if v == "ALLOW":
        return POLICY_PASS
    if v in {"WARN", "REQUIRE_APPROVAL"}:
        return POLICY_WARN
    if v in {"SOFT_BLOCK", "SOURCE_QUARANTINE", "TOOL_FREEZE", "HUMAN_ESCALATE"}:
        return POLICY_BLOCK
    return POLICY_PASS


def _policy_max(a: str, b: str) -> str:
    aa = str(a or POLICY_PASS).upper()
    bb = str(b or POLICY_PASS).upper()
    if POLICY_RANK.get(bb, 0) > POLICY_RANK.get(aa, 0):
        return bb
    return aa


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = str(line).strip()
        if not ln:
            continue
        obj = json.loads(ln)
        if isinstance(obj, dict):
            rows.append(dict(obj))
    return rows


def _load_text(pack_root: Path, relative_path: str, fallback: str) -> str:
    rel = str(relative_path or "").strip()
    if rel:
        p = (pack_root / rel).resolve()
        if p.exists():
            txt = p.read_text(encoding="utf-8").strip()
            if txt:
                return txt
    return str(fallback or "").strip()


def _normalize_policy(v: Any) -> str:
    s = str(v or "").strip().upper()
    if s in POLICY_VALUES:
        return s
    raise ValueError(f"invalid policy value: {v}")


def _normalize_label(v: Any, *, default: str) -> str:
    raw = str(v or "").strip().lower()
    if raw in {"1", "attack", "true"}:
        return "attack"
    if raw in {"0", "benign", "false"}:
        return "benign"
    return str(default)


def _validate_partner_pack_structure(
    *,
    stateless_cases: Sequence[PartnerStatelessCase],
    session_turns: Sequence[PartnerSessionTurn],
) -> None:
    if len(stateless_cases) != 24:
        raise ValueError(f"expected 24 stateless cases, got {len(stateless_cases)}")
    case_ids = [x.case_id for x in stateless_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("duplicate stateless case_id")

    session_case_ids = sorted({x.case_id for x in session_turns})
    if len(session_case_ids) != 24:
        raise ValueError(f"expected 24 unique session case_id values, got {len(session_case_ids)}")
    all_case_ids = set(case_ids) | set(session_case_ids)
    if len(all_case_ids) != 48:
        raise ValueError("case_id must be globally unique across stateless and session")

    stateless_attack = sum(1 for x in stateless_cases if x.family.startswith("attack_"))
    stateless_benign_pass = sum(1 for x in stateless_cases if x.family == "benign_pass")
    stateless_benign_warn = sum(1 for x in stateless_cases if x.family == "benign_warn")
    if stateless_attack != 12 or stateless_benign_pass != 8 or stateless_benign_warn != 4:
        raise ValueError("invalid stateless distribution")

    session_case_family: Dict[str, str] = {}
    for t in session_turns:
        session_case_family.setdefault(t.case_id, t.family)
        if session_case_family[t.case_id] != t.family:
            raise ValueError(f"inconsistent family in session case_id={t.case_id}")
    session_attack = sum(1 for cid, fam in session_case_family.items() if fam.startswith("attack_"))
    session_benign = sum(1 for cid, fam in session_case_family.items() if fam == "benign_long_context")
    if session_attack != 16 or session_benign != 8:
        raise ValueError("invalid session distribution")


def load_partner_pack(
    pack_root: Path,
) -> tuple[List[PartnerStatelessCase], List[PartnerSessionTurn], Dict[str, Any]]:
    manifest_stateless = pack_root / "manifest_stateless.jsonl"
    manifest_session = pack_root / "manifest_session.jsonl"
    meta_path = pack_root / "manifest.meta.json"

    stateless_rows = _read_jsonl(manifest_stateless)
    session_rows = _read_jsonl(manifest_session)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        obj = json.loads(meta_path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            meta = obj

    stateless_cases: List[PartnerStatelessCase] = []
    for row in stateless_rows:
        case = PartnerStatelessCase(
            case_id=str(row.get("case_id", "")).strip(),
            family=str(row.get("family", "")).strip(),
            expected_policy=_normalize_policy(row.get("expected_policy", "")),
            expected_off=bool(row.get("expected_off", False)),
            notes=str(row.get("notes", "")).strip(),
            relative_path=str(row.get("relative_path", "")).strip(),
            text=_load_text(pack_root, str(row.get("relative_path", "")), str(row.get("text", ""))),
        )
        if not case.case_id or not case.family or not case.text:
            raise ValueError(f"invalid stateless case row: {row}")
        stateless_cases.append(case)
    stateless_cases.sort(key=lambda x: x.case_id)

    session_turns: List[PartnerSessionTurn] = []
    for row in session_rows:
        turn = PartnerSessionTurn(
            case_id=str(row.get("case_id", "")).strip(),
            family=str(row.get("family", "")).strip(),
            session_id=str(row.get("session_id", "")).strip(),
            turn_id=int(row.get("turn_id", 0)),
            label_turn=_normalize_label(row.get("label_turn"), default="benign"),
            label_session=_normalize_label(row.get("label_session"), default="benign"),
            expected_policy=_normalize_policy(row.get("expected_policy", "")),
            expected_off=bool(row.get("expected_off", False)),
            notes=str(row.get("notes", "")).strip(),
            relative_path=str(row.get("relative_path", "")).strip(),
            source_ref=str(row.get("source_ref", "")).strip(),
            actor_id=str(row.get("actor_id", "")).strip(),
            bucket=str(row.get("bucket", "core")).strip() or "core",
            eval_slice=str(row.get("eval_slice", "text_intrinsic")).strip() or "text_intrinsic",
            text=_load_text(pack_root, str(row.get("relative_path", "")), str(row.get("text", ""))),
        )
        if (
            not turn.case_id
            or not turn.family
            or not turn.session_id
            or turn.turn_id <= 0
            or not turn.actor_id
            or not turn.text
        ):
            raise ValueError(f"invalid session turn row: {row}")
        session_turns.append(turn)
    session_turns.sort(key=lambda x: (x.case_id, x.turn_id))

    _validate_partner_pack_structure(stateless_cases=stateless_cases, session_turns=session_turns)
    return stateless_cases, session_turns, meta


def _trust_for_label(label: str) -> str:
    return "untrusted" if str(label) == "attack" else "semi"


def _summarize_case_rows(case_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(case_rows)
    pred_counts = defaultdict(int)
    exp_counts = defaultdict(int)
    matches = 0
    stateless_total = 0
    stateless_match = 0
    session_total = 0
    session_match = 0
    for row in case_rows:
        pred = str(row.get("predicted_policy", POLICY_PASS))
        exp = str(row.get("expected_policy", POLICY_PASS))
        pred_counts[pred] += 1
        exp_counts[exp] += 1
        ok = bool(row.get("match", False))
        if ok:
            matches += 1
        ctype = str(row.get("case_type", "stateless"))
        if ctype == "stateless":
            stateless_total += 1
            if ok:
                stateless_match += 1
        else:
            session_total += 1
            if ok:
                session_match += 1
    return {
        "total": int(total),
        "pass": int(pred_counts[POLICY_PASS]),
        "warn": int(pred_counts[POLICY_WARN]),
        "block": int(pred_counts[POLICY_BLOCK]),
        "expected_pass": int(exp_counts[POLICY_PASS]),
        "expected_warn": int(exp_counts[POLICY_WARN]),
        "expected_block": int(exp_counts[POLICY_BLOCK]),
        "match_rate": _safe_div(matches, total),
        "stateless_match_rate": _safe_div(stateless_match, stateless_total),
        "session_match_rate": _safe_div(session_match, session_total),
    }


def _by_family(case_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "total": 0,
        "matches": 0,
        "predicted": defaultdict(int),
        "expected": defaultdict(int),
    })
    for row in case_rows:
        fam = str(row.get("family", "unknown"))
        pred = str(row.get("predicted_policy", POLICY_PASS))
        exp = str(row.get("expected_policy", POLICY_PASS))
        cur = agg[fam]
        cur["total"] += 1
        cur["predicted"][pred] += 1
        cur["expected"][exp] += 1
        if bool(row.get("match", False)):
            cur["matches"] += 1
    out: Dict[str, Any] = {}
    for fam in sorted(agg.keys()):
        cur = agg[fam]
        out[fam] = {
            "total": int(cur["total"]),
            "match_rate": _safe_div(int(cur["matches"]), int(cur["total"])),
            "predicted": {
                POLICY_PASS: int(cur["predicted"][POLICY_PASS]),
                POLICY_WARN: int(cur["predicted"][POLICY_WARN]),
                POLICY_BLOCK: int(cur["predicted"][POLICY_BLOCK]),
            },
            "expected": {
                POLICY_PASS: int(cur["expected"][POLICY_PASS]),
                POLICY_WARN: int(cur["expected"][POLICY_WARN]),
                POLICY_BLOCK: int(cur["expected"][POLICY_BLOCK]),
            },
        }
    return out


def _policy_confusion_matrix(case_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    matrix: Dict[str, Dict[str, int]] = {
        exp: {pred: 0 for pred in (POLICY_PASS, POLICY_WARN, POLICY_BLOCK)}
        for exp in (POLICY_PASS, POLICY_WARN, POLICY_BLOCK)
    }
    for row in case_rows:
        exp = str(row.get("expected_policy", POLICY_PASS))
        pred = str(row.get("predicted_policy", POLICY_PASS))
        if exp not in matrix:
            matrix[exp] = {POLICY_PASS: 0, POLICY_WARN: 0, POLICY_BLOCK: 0}
        if pred not in matrix[exp]:
            matrix[exp][pred] = 0
        matrix[exp][pred] += 1
    return matrix


def evaluate_partner_cases(
    *,
    stateless_cases: Sequence[PartnerStatelessCase],
    session_turns: Sequence[PartnerSessionTurn],
    projector: Any,
    omega_core: OmegaCoreV1,
    off_policy: OffPolicyV1,
) -> Dict[str, Any]:
    case_rows: List[Dict[str, Any]] = []

    for case in sorted(stateless_cases, key=lambda x: x.case_id):
        trust = "semi" if case.family.startswith("benign_") else "untrusted"
        item = ContentItem(
            doc_id=f"partner:{case.case_id}",
            source_id=f"partner_policy_pack:{case.family}",
            source_type="other",
            trust=trust,
            text=case.text,
        )
        proj = projector.project(item)
        state = OmegaState(session_id=f"partner:{case.case_id}", m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        step_result = omega_core.step(state=state, items=[item], projections=[proj])
        decision = off_policy.select_actions(step_result=step_result, items=[item])
        predicted_policy = map_control_outcome_to_partner_policy(str(decision.control_outcome))
        predicted_off = bool(decision.off)
        expected_policy = _normalize_policy(case.expected_policy)
        expected_off = bool(case.expected_off)
        match = (predicted_policy == expected_policy) and (predicted_off == expected_off)
        case_rows.append(
            {
                "case_id": case.case_id,
                "case_type": "stateless",
                "family": case.family,
                "expected_policy": expected_policy,
                "expected_off": expected_off,
                "predicted_policy": predicted_policy,
                "predicted_off": predicted_off,
                "control_outcome": str(decision.control_outcome),
                "severity": str(decision.severity),
                "max_p": float(np.max(step_result.p)) if len(step_result.p) else 0.0,
                "v_sum": float(np.sum(proj.v)),
                "notes": case.notes,
                "relative_path": case.relative_path,
                "match": bool(match),
            }
        )

    grouped: Dict[str, List[PartnerSessionTurn]] = defaultdict(list)
    for row in session_turns:
        grouped[row.case_id].append(row)
    for case_id in sorted(grouped.keys()):
        turns = sorted(grouped[case_id], key=lambda x: int(x.turn_id))
        head = turns[0]
        expected_policy = _normalize_policy(head.expected_policy)
        expected_off = bool(head.expected_off)

        state = OmegaState(session_id=head.session_id, m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        policy_max = POLICY_PASS
        off_any = False
        turn_policy_hist = {POLICY_PASS: 0, POLICY_WARN: 0, POLICY_BLOCK: 0}
        first_block_turn: Optional[int] = None
        max_p_seen = 0.0

        for turn in turns:
            item = ContentItem(
                doc_id=f"partner:{turn.case_id}:t{int(turn.turn_id):02d}",
                source_id=turn.source_ref or f"partner_policy_pack:{turn.family}",
                source_type="other",
                trust=_trust_for_label(turn.label_turn),
                text=turn.text,
            )
            proj = projector.project(item)
            step_result = omega_core.step(state=state, items=[item], projections=[proj])
            decision = off_policy.select_actions(step_result=step_result, items=[item])
            state = OmegaState(
                session_id=head.session_id,
                m=np.array(step_result.m_next, dtype=float),
                step=int(state.step) + 1,
            )

            cur_policy = map_control_outcome_to_partner_policy(str(decision.control_outcome))
            policy_max = _policy_max(policy_max, cur_policy)
            turn_policy_hist[cur_policy] += 1
            off_any = bool(off_any or bool(decision.off))
            max_p_seen = max(max_p_seen, float(np.max(step_result.p)) if len(step_result.p) else 0.0)
            if cur_policy == POLICY_BLOCK and first_block_turn is None:
                first_block_turn = int(turn.turn_id)

        match = (policy_max == expected_policy) and (off_any == expected_off)
        case_rows.append(
            {
                "case_id": head.case_id,
                "case_type": "session",
                "family": head.family,
                "expected_policy": expected_policy,
                "expected_off": expected_off,
                "predicted_policy": policy_max,
                "predicted_off": bool(off_any),
                "control_outcome": policy_max,
                "severity": "SESSION",
                "max_p": float(max_p_seen),
                "v_sum": 0.0,
                "notes": head.notes,
                "relative_path": "",
                "session_id": head.session_id,
                "actor_id": head.actor_id,
                "turn_count": int(len(turns)),
                "first_block_turn": int(first_block_turn) if first_block_turn is not None else None,
                "turn_policy_histogram": dict(turn_policy_hist),
                "match": bool(match),
            }
        )

    case_rows.sort(key=lambda x: (str(x.get("case_type", "")), str(x.get("case_id", ""))))
    mismatch_cases = [row for row in case_rows if not bool(row.get("match", False))]
    return {
        "case_rows": case_rows,
        "mismatch_cases": mismatch_cases,
        "summary": _summarize_case_rows(case_rows),
        "by_family": _by_family(case_rows),
        "policy_confusion_matrix": _policy_confusion_matrix(case_rows),
    }


def _baseline_compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    cur = current.get("summary", {}) if isinstance(current.get("summary"), Mapping) else {}
    base = baseline.get("summary", {}) if isinstance(baseline.get("summary"), Mapping) else {}
    keys_float = ("match_rate", "stateless_match_rate", "session_match_rate")
    keys_count = ("pass", "warn", "block")
    delta: Dict[str, Any] = {}
    for k in keys_float:
        delta[k] = float(cur.get(k, 0.0)) - float(base.get(k, 0.0))
    for k in keys_count:
        delta[k] = int(cur.get(k, 0)) - int(base.get(k, 0))
    return {"summary_delta": delta}


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate deterministic partner policy pack (PASS/WARN/BLOCK).")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--mode", choices=["pi0", "hybrid"], default="pi0")
    parser.add_argument("--pack-root", default="tests/data/partner_policy_pack/v1")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--artifacts-root", default="artifacts/partner_policy_eval")
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--warn-enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-approval-on-warn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warn-max-p-gte", type=float, default=0.45)
    parser.add_argument("--warn-sum-m-next-gte", type=float, default=0.35)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    pack_root = (ROOT / str(args.pack_root)).resolve()
    stateless_cases, session_turns, pack_meta = load_partner_pack(pack_root)

    snapshot = load_resolved_config(
        profile=str(args.profile),
        cli_overrides={"projector": {"mode": str(args.mode)}},
    )
    cfg = deepcopy(snapshot.resolved)
    off_cfg = cfg.setdefault("off_policy", {})
    if not isinstance(off_cfg, dict):
        off_cfg = {}
        cfg["off_policy"] = off_cfg
    control_cfg = off_cfg.setdefault("control_outcome", {})
    if not isinstance(control_cfg, dict):
        control_cfg = {}
        off_cfg["control_outcome"] = control_cfg
    warn_cfg = control_cfg.setdefault("warn", {})
    if not isinstance(warn_cfg, dict):
        warn_cfg = {}
    warn_cfg["enabled"] = bool(args.warn_enabled)
    warn_cfg["max_p_gte"] = float(args.warn_max_p_gte)
    warn_cfg["sum_m_next_gte"] = float(args.warn_sum_m_next_gte)
    control_cfg["warn"] = warn_cfg
    req_cfg = control_cfg.setdefault("require_approval", {})
    if not isinstance(req_cfg, dict):
        req_cfg = {}
    req_cfg["enabled"] = bool(args.require_approval_on_warn)
    req_cfg["on_warn"] = bool(args.require_approval_on_warn)
    req_cfg["on_off"] = bool(req_cfg.get("on_off", False))
    control_cfg["require_approval"] = req_cfg
    off_cfg["control_outcome"] = control_cfg
    cfg["off_policy"] = off_cfg

    projector = build_projector(cfg)
    omega_core = OmegaCoreV1(omega_params_from_config(cfg))
    off_policy = OffPolicyV1(cfg)

    eval_block = evaluate_partner_cases(
        stateless_cases=stateless_cases,
        session_turns=session_turns,
        projector=projector,
        omega_core=omega_core,
        off_policy=off_policy,
    )

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"partner_policy_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_path = out_dir / "rows.jsonl"
    mismatches_path = out_dir / "mismatches.json"
    report_path = out_dir / "report.json"
    _write_jsonl(rows_path, eval_block["case_rows"])
    mismatches_path.write_text(json.dumps(eval_block["mismatch_cases"], ensure_ascii=False, indent=2), encoding="utf-8")

    baseline_compare = None
    if args.baseline_report:
        baseline_path = (ROOT / str(args.baseline_report)).resolve()
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            if isinstance(baseline, dict):
                baseline_compare = {"path": str(baseline_path), **_baseline_compare(eval_block, baseline)}

    report: Dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "profile": str(args.profile),
        "mode": str(args.mode),
        "seed": int(args.seed),
        "pack_root": str(pack_root),
        "summary": eval_block["summary"],
        "by_family": eval_block["by_family"],
        "mismatch_cases": eval_block["mismatch_cases"],
        "policy_confusion_matrix": eval_block["policy_confusion_matrix"],
        "pack_meta": pack_meta,
        "policy_mapping": {
            "PASS": "ALLOW",
            "WARN": ["WARN", "REQUIRE_APPROVAL"],
            "BLOCK": ["SOFT_BLOCK", "SOURCE_QUARANTINE", "TOOL_FREEZE", "HUMAN_ESCALATE"],
        },
        "runtime_overrides": {
            "warn_enabled": bool(args.warn_enabled),
            "require_approval_on_warn": bool(args.require_approval_on_warn),
            "warn_max_p_gte": float(args.warn_max_p_gte),
            "warn_sum_m_next_gte": float(args.warn_sum_m_next_gte),
        },
        "baseline_compare": baseline_compare,
        "artifacts": {
            "report_json": str(report_path),
            "rows_jsonl": str(rows_path),
            "mismatches_json": str(mismatches_path),
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
