from __future__ import annotations

from scripts.run_release_gate import evaluate_gates


def _base_metrics() -> dict:
    return {
        "commands": {
            "pytest": {"exit_code": 0},
            "run_eval": {"exit_code": 0},
            "run_local_contour": {"exit_code": 0},
            "smoke_real_rag": {"exit_code": 0},
        },
        "run_eval": {
            "hard_negatives": {"fp": 0},
            "whitebox": {"bypass_rate": 0.10, "base_detect_rate": 0.97},
        },
        "run_local_contour": {
            "fp_horizon_step": None,
            "benign_off_count": 0,
            "steps_to_off_sla_violations": 0,
            "tool_freeze_enforcement_rate": 1.0,
            "tool_gateway_coverage": 1.0,
            "orphan_tool_executions": 0,
        },
        "smoke_real_rag": {
            "failures_count": 0,
            "gateway_coverage_min": 1.0,
            "bypass_count": 0,
        },
    }


def _gates() -> list[dict]:
    return [
        {"id": "RG-001", "metric": "commands.pytest.exit_code", "op": "eq", "threshold": 0},
        {"id": "RG-006", "metric": "run_eval.whitebox.bypass_rate", "op": "le", "threshold": 0.20},
        {"id": "RG-007", "metric": "run_eval.whitebox.base_detect_rate", "op": "ge", "threshold": 0.95},
        {"id": "RG-008", "metric": "run_local_contour.fp_horizon_step", "op": "is_null", "threshold": None},
        {"id": "RG-012", "metric": "run_local_contour.tool_gateway_coverage", "op": "eq", "threshold": 1.0},
        {"id": "RG-013", "metric": "run_local_contour.orphan_tool_executions", "op": "eq", "threshold": 0},
    ]


def test_release_gate_eval_passes_on_valid_metrics():
    results = evaluate_gates(_base_metrics(), _gates())
    assert all(r["status"] == "PASS" for r in results)


def test_release_gate_eval_fails_on_whitebox_bypass():
    metrics = _base_metrics()
    metrics["run_eval"]["whitebox"]["bypass_rate"] = 0.31
    results = evaluate_gates(metrics, _gates())
    failed = {r["id"] for r in results if r["status"] == "FAIL"}
    assert "RG-006" in failed


def test_release_gate_eval_fails_on_fp_horizon_and_coverage_and_orphans():
    metrics = _base_metrics()
    metrics["run_local_contour"]["fp_horizon_step"] = 7
    metrics["run_local_contour"]["tool_gateway_coverage"] = 0.75
    metrics["run_local_contour"]["orphan_tool_executions"] = 1
    results = evaluate_gates(metrics, _gates())
    failed = {r["id"] for r in results if r["status"] == "FAIL"}
    assert "RG-008" in failed
    assert "RG-012" in failed
    assert "RG-013" in failed
