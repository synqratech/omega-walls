from __future__ import annotations

from scripts.replay_incident import (
    compare_incident_artifacts,
    compare_replay_results,
    compare_snapshots,
    diff_policy_outputs,
    extract_incident_snapshot,
    parse_replay_input,
)


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


def test_parse_replay_input_normalizes_single_turn_shape():
    payload = {
        "event": "omega_replay_input_v1",
        "schema_version": "1.0",
        "session_id": "sess-1",
        "actor_id": "actor-1",
        "turn_index": 1,
        "user_query": "Summarize safe text",
        "packet_items": [
            {
                "doc_id": "doc-1",
                "source_id": "source-1",
                "source_type": "retrieved_chunk",
                "trust": "trusted",
                "text": "safe content",
            }
        ],
        "tool_requests": [{"tool_name": "summarize", "args": {"k": 2}}],
    }
    parsed = parse_replay_input(payload)
    assert parsed["event"] == "omega_replay_input_v1"
    assert len(parsed["turns"]) == 1
    turn = parsed["turns"][0]
    assert turn["session_id"] == "sess-1"
    assert turn["actor_id"] == "actor-1"
    assert turn["packet_items"][0]["text"] == "safe content"
    assert turn["tool_requests"][0]["tool_name"] == "summarize"


def test_parse_replay_input_rejects_missing_packet_text():
    payload = {
        "event": "omega_replay_input_v1",
        "turns": [
            {
                "session_id": "sess-1",
                "actor_id": "actor-1",
                "turn_index": 1,
                "user_query": "Q",
                "packet_items": [
                    {
                        "doc_id": "doc-1",
                        "source_id": "source-1",
                        "source_type": "retrieved_chunk",
                        "trust": "trusted",
                    }
                ],
            }
        ],
    }
    try:
        parse_replay_input(payload)
    except ValueError as exc:
        assert ".text is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing packet_items.text")


def test_compare_replay_results_detects_incident_delta():
    base_turn = {
        "turn_index": 1,
        "control_outcome": "WARN",
        "off": False,
        "severity": "L1",
        "reason_flags": ["reason_wall"],
        "action_types": ["WARN"],
        "actions": [{"type": "WARN", "target": "SESSION", "doc_ids": [], "source_ids": [], "tool_mode": None, "allowlist": [], "horizon_steps": None}],
        "top_docs": ["doc-1"],
        "tool_decisions": [],
        "tool_executions": [],
        "blocked_doc_ids": [],
        "quarantined_source_ids": [],
        "prevented_tools": [],
        "incident_30s": {"why": {"control_outcome": "WARN"}},
        "incident_artifact": {"decision": {"control_outcome": "WARN"}},
    }
    actual = {"turns": [base_turn], "signature": "a"}
    expected = {"turns": [{**base_turn, "incident_30s": {"why": {"control_outcome": "TOOL_FREEZE"}}}], "signature": "b"}
    cmp = compare_replay_results(actual, expected)
    assert cmp["match"] is False
    assert cmp["components"]["turns"][0]["match"] is False
    assert "incident_30s" in cmp["components"]["turns"][0]["changed_components"]


def test_diff_policy_outputs_no_decision_delta_flag():
    turn = {
        "turn_index": 1,
        "control_outcome": "ALLOW",
        "off": False,
        "severity": "L1",
        "reason_flags": [],
        "action_types": [],
        "actions": [],
        "top_docs": [],
        "tool_decisions": [],
        "tool_executions": [],
        "blocked_doc_ids": [],
        "quarantined_source_ids": [],
        "prevented_tools": [],
        "incident_30s": {},
    }
    diff = diff_policy_outputs({"turns": [turn]}, {"turns": [dict(turn)]})
    assert diff["no_decision_delta"] is True
    assert diff["turn_deltas"] == []


def test_compare_incident_artifacts_detects_id_mismatch():
    actual = {
        "trace_id": "trc_111",
        "decision_id": "dec_111",
        "context": {"session_id": "s1", "step": 1},
        "decision": {"control_outcome": "WARN"},
        "reasons": {"reason_flags": ["reason_wall"]},
        "sources": {"top_docs": []},
        "prevention": {"context_prevented": False},
    }
    expected = {
        "trace_id": "trc_111",
        "decision_id": "dec_222",
        "context": {"session_id": "s1", "step": 1},
        "decision": {"control_outcome": "WARN"},
        "reasons": {"reason_flags": ["reason_wall"]},
        "sources": {"top_docs": []},
        "prevention": {"context_prevented": False},
    }
    cmp = compare_incident_artifacts(actual, expected)
    assert cmp["match"] is False
    assert cmp["components"]["ids"]["trace_id"] is True
    assert cmp["components"]["ids"]["decision_id"] is False
