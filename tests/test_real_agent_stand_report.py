from __future__ import annotations

import scripts.run_real_agent_stand as stand


def test_real_agent_stand_gate_aggregation() -> None:
    phase1 = {
        "status": "ok",
        "report": {
            "summary": {
                "blocked_input_seen": True,
                "blocked_tool_seen": True,
                "require_approval_seen": True,
                "outage_fail_closed_seen": True,
                "orphan_executions_zero": True,
                "gateway_coverage_ok": True,
            }
        },
    }
    phase2 = {
        "status": "ok",
        "smoke_payload": {
            "sample_block_decision": {"block": True},
            "sample_require_approval_decision": {"requireApproval": True},
            "webfetch_guard_seen": True,
        },
        "local_api_payload": {"session_reset_seen": True},
    }
    gates = stand._gates(phase1, phase2)  # noqa: SLF001
    assert gates["blocked_input_seen"] is True
    assert gates["blocked_tool_seen"] is True
    assert gates["require_approval_seen"] is True
    assert gates["webfetch_guard_seen"] is True
    assert gates["outage_fail_closed_seen"] is True
    assert gates["orphan_executions_zero"] is True
    assert gates["gateway_coverage_ok"] is True
    assert gates["session_reset_seen"] is True


def test_real_agent_stand_status_aggregation_matrix() -> None:
    ok_gates = {
        "blocked_input_seen": True,
        "blocked_tool_seen": True,
        "require_approval_seen": True,
        "webfetch_guard_seen": True,
        "outage_fail_closed_seen": True,
        "orphan_executions_zero": True,
        "gateway_coverage_ok": True,
        "session_reset_seen": True,
    }
    assert (
        stand.aggregate_status(
            strict=True,
            phase1_status="ok",
            phase2_status="ok",
            gates=ok_gates,
            hard_errors=[],
        )
        == "ok"
    )
    partial = dict(ok_gates)
    partial["webfetch_guard_seen"] = False
    assert (
        stand.aggregate_status(
            strict=False,
            phase1_status="ok",
            phase2_status="ok",
            gates=partial,
            hard_errors=[],
        )
        == "partial"
    )
    assert (
        stand.aggregate_status(
            strict=True,
            phase1_status="ok",
            phase2_status="ok",
            gates=partial,
            hard_errors=[],
        )
        == "fail"
    )
    assert (
        stand.aggregate_status(
            strict=False,
            phase1_status="ok",
            phase2_status="ok",
            gates=ok_gates,
            hard_errors=["api_boot_failure"],
        )
        == "fail"
    )
