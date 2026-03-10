from __future__ import annotations

from scripts.replay_incident import compare_snapshots, extract_incident_snapshot


def test_replay_extract_local_contour_snapshot():
    payload = {
        "scenarios": [
            {
                "scenario": "attack_case",
                "steps": [
                    {
                        "step": 1,
                        "off": False,
                        "reasons": {"reason_spike": False, "reason_wall": False, "reason_sum": False, "reason_multi": False},
                        "actions": [],
                        "top_docs": [],
                        "tool_decisions": [],
                    },
                    {
                        "step": 2,
                        "off": True,
                        "reasons": {"reason_spike": True, "reason_wall": True, "reason_sum": True, "reason_multi": False},
                        "actions": [{"type": "TOOL_FREEZE", "target": "TOOLS", "allowlist": ["summarize"]}],
                        "top_docs": ["doc-1"],
                        "tool_decisions": [{"allowed": False, "mode": "TOOLS_DISABLED", "reason": "OFF_STATE_BLOCK"}],
                    },
                ],
            }
        ]
    }
    snap = extract_incident_snapshot(payload, report_kind="local_contour", scenario="attack_case", step=2)
    assert snap["step"] == 2
    assert snap["off"] is True
    assert snap["top_docs"] == ["doc-1"]
    assert snap["actions"][0]["type"] == "TOOL_FREEZE"


def test_replay_compare_detects_mismatch():
    actual = {
        "off": True,
        "reasons": {"reason_spike": True},
        "actions": [{"type": "TOOL_FREEZE", "target": "TOOLS", "doc_ids": [], "source_ids": [], "tool_mode": None, "allowlist": [], "horizon_steps": None}],
        "top_docs": ["doc-a"],
        "tool_decisions": [{"allowed": False, "mode": "TOOLS_DISABLED", "reason": "OFF_STATE_BLOCK", "logged": True}],
    }
    expected = {
        "off": True,
        "reasons": {"reason_spike": True},
        "actions": [{"type": "TOOL_FREEZE", "target": "TOOLS", "doc_ids": [], "source_ids": [], "tool_mode": None, "allowlist": [], "horizon_steps": None}],
        "top_docs": ["doc-b"],
        "tool_decisions": [{"allowed": False, "mode": "TOOLS_DISABLED", "reason": "OFF_STATE_BLOCK", "logged": True}],
    }
    cmp = compare_snapshots(actual, expected)
    assert cmp["match"] is False
    assert cmp["components"]["top_docs"] is False
    assert cmp["components"]["reasons"] is True
