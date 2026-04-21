from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from omega import cli as omega_cli


def _write_events(path: Path) -> None:
    rows = [
        {
            "ts": "2026-04-16T10:00:00Z",
            "surface": "sdk",
            "session_id": "sess-1",
            "actor_id": "sess-1",
            "mode": "monitor",
            "risk_score": 0.25,
            "intended_action": "ALLOW",
            "actual_action": "ALLOW",
            "triggered_rules": [],
            "attribution": [],
            "reason_codes": [],
            "trace_id": "tr-1",
            "decision_id": "dc-1",
        },
        {
            "ts": "2026-04-16T10:00:30Z",
            "surface": "sdk",
            "session_id": "sess-1",
            "actor_id": "sess-1",
            "mode": "monitor",
            "risk_score": 0.83,
            "intended_action": "SOFT_BLOCK",
            "actual_action": "ALLOW",
            "triggered_rules": ["override_instructions"],
            "reason_codes": ["reason_spike"],
            "trace_id": "tr-2",
            "decision_id": "dc-2",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def test_cli_explain_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    _write_events(events_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["omega-walls", "explain", "--session", "sess-1", "--events-path", str(events_path), "--format", "json"],
    )
    omega_cli.main()
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["session_id"] == "sess-1"
    assert payload["summary"]["events_count"] == 2


def test_cli_explain_csv(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    _write_events(events_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["omega-walls", "explain", "--session", "sess-1", "--events-path", str(events_path), "--format", "csv"],
    )
    omega_cli.main()
    out = capsys.readouterr().out
    assert "session_id,index,ts,surface,risk_score" in out
    assert "sess-1,2,2026-04-16T10:00:30Z" in out


def test_cli_explain_missing_session_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    _write_events(events_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["omega-walls", "explain", "--session", "missing", "--events-path", str(events_path), "--format", "json"],
    )
    with pytest.raises(SystemExit) as exc:
        omega_cli.main()
    err = capsys.readouterr().err
    assert int(exc.value.code) == 2
    assert "no monitor events found for session" in err
