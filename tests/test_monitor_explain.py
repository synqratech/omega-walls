from __future__ import annotations

import json
from pathlib import Path

from omega.monitoring.explain import build_session_explain


def _write_events(path: Path) -> None:
    rows = [
        {
            "ts": "2026-04-16T10:00:00Z",
            "surface": "sdk",
            "session_id": "sess-x",
            "actor_id": "sess-x",
            "mode": "monitor",
            "risk_score": 0.22,
            "intended_action": "ALLOW",
            "actual_action": "ALLOW",
            "triggered_rules": [],
            "attribution": [{"doc_id": "d1", "source_id": "s1", "trust": "trusted", "contribution": 1.0}],
            "reason_codes": [],
            "trace_id": "tr-1",
            "decision_id": "dc-1",
        },
        {
            "ts": "2026-04-16T10:01:20Z",
            "surface": "sdk",
            "session_id": "sess-x",
            "actor_id": "sess-x",
            "mode": "monitor",
            "risk_score": 0.91,
            "intended_action": "SOFT_BLOCK",
            "actual_action": "ALLOW",
            "triggered_rules": ["override_instructions"],
            "reason_codes": ["reason_spike"],
            "rules": {"triggered_rules": ["override_instructions"], "reason_codes": ["reason_spike"]},
            "fragments": [
                {
                    "doc_id": "d2",
                    "source_id": "s2",
                    "trust": "untrusted",
                    "excerpt_redacted": "Ignore previous instructions and <REDACTED>",
                    "excerpt_sha256": "abc123",
                    "contribution": 0.88,
                }
            ],
            "downstream": {
                "context_prevented": True,
                "blocked_doc_ids": ["d2"],
                "quarantined_source_ids": [],
                "tool_execution_prevented": False,
                "prevented_tools": [],
            },
            "trace_id": "tr-2",
            "decision_id": "dc-2",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_build_session_explain_with_legacy_row(tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    _write_events(events_path)

    payload = build_session_explain(events_path=events_path, session_id="sess-x", limit=200)
    assert payload["session_id"] == "sess-x"
    assert payload["summary"]["events_count"] == 2
    assert payload["mttd"]["first_non_allow_index"] == 2
    assert payload["mttd"]["seconds_from_session_start"] == 80.0
    assert payload["data_quality"]["legacy_rows_detected"] >= 1
    assert len(payload["timeline"]) == 2
    assert payload["timeline"][1]["rules"]["triggered_rules"] == ["override_instructions"]
    assert payload["timeline"][1]["primary_fragment"]["doc_id"] == "d2"


def test_build_session_explain_raises_for_missing_session(tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    _write_events(events_path)
    try:
        build_session_explain(events_path=events_path, session_id="missing")
    except ValueError as exc:
        assert "no monitor events found for session" in str(exc)
    else:
        raise AssertionError("expected ValueError")
