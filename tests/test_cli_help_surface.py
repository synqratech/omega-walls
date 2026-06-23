from __future__ import annotations

import json

import pytest

import omega.cli as cli


def test_root_help_lists_key_subcommands(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["omega-walls", "--help"])
    cli.main()
    out = capsys.readouterr().out
    for token in ("analyze", "report", "explain", "telemetry", "alerts", "fallback", "orchestrator", "replay", "keys"):
        assert token in out


def test_no_args_shows_root_help_and_does_not_crash(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli.sys, "argv", ["omega-walls"])
    cli.main()
    out = capsys.readouterr().out
    assert "Omega Walls CLI" in out
    assert "Subcommands" in out


def test_legacy_analyze_invocation_still_works(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(_args: object) -> dict[str, object]:
        return {"status": "ok", "legacy": True}

    monkeypatch.setattr(cli, "_run_analyze", _fake_run)
    monkeypatch.setattr(cli.sys, "argv", ["omega-walls", "--profile", "quickstart", "--text", "hi"])
    cli.main()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["legacy"] is True
