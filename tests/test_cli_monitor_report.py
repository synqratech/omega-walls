from __future__ import annotations

import json
import sys
from pathlib import Path

from omega import cli as omega_cli


def _write_events(path: Path) -> None:
    rows = [
        {
            "ts": "2026-04-16T10:00:00Z",
            "surface": "sdk",
            "session_id": "s1",
            "actor_id": "s1",
            "mode": "monitor",
            "risk_score": 0.82,
            "intended_action": "SOFT_BLOCK",
            "actual_action": "ALLOW",
            "triggered_rules": ["override_instructions"],
            "attribution": [],
            "reason_codes": ["reason_spike"],
            "trace_id": "trc_1",
            "decision_id": "dec_1",
        },
        {
            "ts": "2026-04-16T10:01:00Z",
            "surface": "sdk",
            "session_id": "s2",
            "actor_id": "s2",
            "mode": "monitor",
            "risk_score": 0.24,
            "intended_action": "ALLOW",
            "actual_action": "ALLOW",
            "triggered_rules": [],
            "attribution": [],
            "reason_codes": [],
            "trace_id": "trc_2",
            "decision_id": "dec_2",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_cli_report_json_output(monkeypatch, capsys, tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    _write_events(events_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["omega-walls", "report", "--events-path", str(events_path), "--format", "json"],
    )
    omega_cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["total_checks"] == 2
    assert payload["would_block"] == 1


def test_cli_report_csv_output(monkeypatch, capsys, tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    _write_events(events_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["omega-walls", "report", "--events-path", str(events_path), "--format", "csv"],
    )
    omega_cli.main()
    out = capsys.readouterr().out
    assert "key,value" in out
    assert "total_checks,2" in out
