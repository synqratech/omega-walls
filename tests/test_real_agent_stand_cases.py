from __future__ import annotations

import scripts.smoke_langchain_real_workflow as phase1


def test_langchain_real_workflow_suite_contract() -> None:
    cases = list(phase1.DEFAULT_CASES)
    assert len(cases) == 10
    ids = [case.case_id for case in cases]
    assert len(set(ids)) == 10
    assert ids[0] == "lc01_benign_allow"
    assert ids[-1] == "lc10_continuity_benign_allow"
    expected_categories = {case.expected_category for case in cases}
    assert {
        "allow_input",
        "blocked_input",
        "blocked_tool",
        "escalation_like",
        "allow_tool_once",
        "outage_blocked_before_tool",
    }.issubset(expected_categories)


def test_langchain_real_workflow_summary_logic() -> None:
    rows = [
        {
            "case_id": "a",
            "observed_category": "allow_input",
            "passed": True,
            "gateway_coverage": 1.0,
            "orphan_executions": 0,
            "require_approval_seen": False,
        },
        {
            "case_id": "b",
            "observed_category": "blocked_input",
            "passed": True,
            "gateway_coverage": 1.0,
            "orphan_executions": 0,
            "require_approval_seen": True,
        },
        {
            "case_id": "c",
            "observed_category": "blocked_tool",
            "passed": True,
            "gateway_coverage": 1.0,
            "orphan_executions": 0,
            "require_approval_seen": False,
        },
        {
            "case_id": "d",
            "observed_category": "outage_blocked_before_tool",
            "passed": True,
            "gateway_coverage": 1.0,
            "orphan_executions": 0,
            "require_approval_seen": False,
        },
    ]
    summary = phase1.summarize_case_results(rows)
    assert summary["total_cases"] == 4
    assert summary["failed_cases"] == 0
    assert summary["blocked_input_seen"] is True
    assert summary["blocked_tool_seen"] is True
    assert summary["require_approval_seen"] is True
    assert summary["outage_fail_closed_seen"] is True
    assert summary["gateway_coverage_ok"] is True
    assert summary["orphan_executions_zero"] is True
