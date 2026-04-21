from __future__ import annotations

from scripts.eval_session_pi_gate import SessionOutcome
from scripts.eval_session_stateful_focus_gate import (
    _focus_baseline_delta,
    build_focus_summary,
    evaluate_focus_gate,
)


def _mk(
    *,
    sid: str,
    bucket: str,
    family: str,
    label: str,
    off: bool,
    first_off_turn: int | None = 1,
) -> SessionOutcome:
    return SessionOutcome(
        session_id=sid,
        actor_id=f"actor:{sid}",
        bucket=bucket,
        family=family,
        label_session=label,
        eval_slice="text_intrinsic",
        turn_count=4,
        detected_off=bool(off),
        first_off_turn=(int(first_off_turn) if off and first_off_turn is not None else None),
        late_detect=False,
        max_turn_p=1.0 if off else 0.0,
        off_reasons={},
    )


def test_build_focus_summary_splits_core_and_cross():
    core = [
        _mk(sid="c1", bucket="core", family="cocktail", label="attack", off=True),
        _mk(sid="c2", bucket="core", family="distributed_wo_explicit", label="attack", off=False),
        _mk(sid="c3", bucket="core", family="benign_long_context", label="benign", off=False),
    ]
    cross = [
        _mk(sid="x1", bucket="cross_session", family="cocktail", label="attack", off=True),
        _mk(sid="x2", bucket="cross_session", family="distributed_wo_explicit", label="attack", off=True),
    ]
    out = build_focus_summary(core_outcomes=core, cross_outcomes=cross)
    assert out["core_all"]["sessions_total"] == 3
    assert out["cross_all"]["sessions_total"] == 2
    assert out["focus_all"]["sessions_total"] == 5
    assert out["core_cocktail_attack"]["attack_sessions"] == 1
    assert out["cross_distributed_attack"]["attack_sessions"] == 1


def test_focus_gate_pass_and_fail():
    summary_ok = {
        "core_all": {"session_benign_off_rate": 0.0},
        "core_cocktail_attack": {"session_attack_off_rate": 1.0},
        "core_distributed_attack": {"session_attack_off_rate": 0.95},
        "cross_cocktail_attack": {"session_attack_off_rate": 1.0},
        "cross_distributed_attack": {"session_attack_off_rate": 0.95},
        "cross_all": {"session_attack_off_rate": 0.95},
    }
    gate_ok = evaluate_focus_gate(
        summary_block=summary_ok,
        core_benign_le=0.0111,
        core_cocktail_attack_ge=0.97,
        core_distributed_attack_ge=0.90,
        cross_cocktail_attack_ge=0.90,
        cross_distributed_attack_ge=0.90,
        cross_all_attack_ge=0.92,
    )
    assert gate_ok["passed"] is True

    summary_bad = {
        "core_all": {"session_benign_off_rate": 0.02},
        "core_cocktail_attack": {"session_attack_off_rate": 0.96},
        "core_distributed_attack": {"session_attack_off_rate": 0.89},
        "cross_cocktail_attack": {"session_attack_off_rate": 0.85},
        "cross_distributed_attack": {"session_attack_off_rate": 0.88},
        "cross_all": {"session_attack_off_rate": 0.90},
    }
    gate_bad = evaluate_focus_gate(
        summary_block=summary_bad,
        core_benign_le=0.0111,
        core_cocktail_attack_ge=0.97,
        core_distributed_attack_ge=0.90,
        cross_cocktail_attack_ge=0.90,
        cross_distributed_attack_ge=0.90,
        cross_all_attack_ge=0.92,
    )
    assert gate_bad["passed"] is False
    assert any(c["status"] == "FAIL" for c in gate_bad["checks"])


def test_focus_baseline_delta_fields():
    cur = {
        "core_all": {"session_attack_off_rate": 0.99, "session_benign_off_rate": 0.0},
        "cross_all": {"session_attack_off_rate": 0.94},
        "core_cocktail_attack": {"session_attack_off_rate": 1.0},
        "core_distributed_attack": {"session_attack_off_rate": 0.95},
        "cross_cocktail_attack": {"session_attack_off_rate": 0.93},
        "cross_distributed_attack": {"session_attack_off_rate": 0.95},
    }
    base = {
        "core_all": {"session_attack_off_rate": 0.98, "session_benign_off_rate": 0.01},
        "cross_all": {"session_attack_off_rate": 0.92},
        "core_cocktail_attack": {"session_attack_off_rate": 0.98},
        "core_distributed_attack": {"session_attack_off_rate": 0.90},
        "cross_cocktail_attack": {"session_attack_off_rate": 0.90},
        "cross_distributed_attack": {"session_attack_off_rate": 0.92},
    }
    out = _focus_baseline_delta(cur, base)
    d = out["metric_delta"]
    assert abs(float(d["core_all.session_attack_off_rate"]) - 0.01) < 1e-9
    assert abs(float(d["core_all.session_benign_off_rate"]) - (-0.01)) < 1e-9
