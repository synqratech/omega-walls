from __future__ import annotations

import json
from pathlib import Path

from scripts.check_iteration_gates import _resolve_report_path, evaluate_iteration_gates


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def test_resolve_report_path_with_wrapper(tmp_path: Path) -> None:
    report = tmp_path / "real_report.json"
    wrapper = tmp_path / "BASELINE_LATEST.json"
    _write_json(report, {"summary": {"attack_off_rate": 0.4, "benign_off_rate": 0.1}})
    _write_json(wrapper, {"report": str(report)})
    resolved = _resolve_report_path(str(wrapper))
    assert resolved == report


def test_evaluate_iteration_gates_pass() -> None:
    promptshield = {"summary": {"attack_off_rate": 0.44, "benign_off_rate": 0.06}}
    session = {
        "summary_core_text_intrinsic": {"session_benign_off_rate": 0.01, "session_attack_off_rate": 1.0},
        "cross_session": {"session_attack_off_rate": 0.93},
    }
    wainject = {"summary": {"attack_off_rate": 0.48, "benign_off_rate": 0.008}}
    required, targets = evaluate_iteration_gates(
        promptshield_report=promptshield,
        session_report=session,
        wainject_report=wainject,
        promptshield_base_attack=0.3996,
        promptshield_base_benign=0.1026,
        wainject_base_attack=0.4914,
    )
    assert all(g.status == "PASS" for g in required)
    assert all(g.status == "PASS" for g in targets)


def test_evaluate_iteration_gates_required_fail() -> None:
    promptshield = {"summary": {"attack_off_rate": 0.395, "benign_off_rate": 0.09}}
    session = {
        "summary_core_text_intrinsic": {"session_benign_off_rate": 0.02, "session_attack_off_rate": 0.98},
        "cross_session": {"session_attack_off_rate": 0.91},
    }
    wainject = {"summary": {"attack_off_rate": 0.46, "benign_off_rate": 0.0101}}
    required, _targets = evaluate_iteration_gates(
        promptshield_report=promptshield,
        session_report=session,
        wainject_report=wainject,
        promptshield_base_attack=0.3996,
        promptshield_base_benign=0.1026,
        wainject_base_attack=0.4914,
    )
    failed = [g.id for g in required if g.status == "FAIL"]
    assert "PS-REQ-BENIGN-30PCT" in failed
    assert "SES-REQ-CORE-BENIGN" in failed
    assert "SES-REQ-CORE-ATTACK" in failed
    assert "SES-REQ-CROSS-ATTACK" in failed
    assert "WAI-REQ-BENIGN" in failed


def test_evaluate_iteration_gates_prefers_cross_actor_group() -> None:
    promptshield = {"summary": {"attack_off_rate": 0.44, "benign_off_rate": 0.06}}
    session = {
        "summary_core_text_intrinsic": {"session_benign_off_rate": 0.01, "session_attack_off_rate": 1.0},
        "cross_session": {"session_attack_off_rate": 0.40},
        "cross_session_actor_group": {"session_attack_off_rate": 0.95},
    }
    wainject = {"summary": {"attack_off_rate": 0.48, "benign_off_rate": 0.008}}
    required, _targets = evaluate_iteration_gates(
        promptshield_report=promptshield,
        session_report=session,
        wainject_report=wainject,
        promptshield_base_attack=0.3996,
        promptshield_base_benign=0.1026,
        wainject_base_attack=0.4914,
    )
    cross_gate = next(g for g in required if g.id == "SES-REQ-CROSS-ATTACK")
    assert cross_gate.status == "PASS"
    assert cross_gate.metric == "session.cross_session_actor_group.session_attack_off_rate"
