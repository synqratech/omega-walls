from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import pytest

from omega.cli import _run_telemetry


def _tmp_dir(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_telemetry_cli_status_show_pending_disable(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = _tmp_dir("telemetry_cli")
    cfg = {
        "telemetry": {
            "enabled": True,
            "endpoint": "https://telemetry.omega-walls.io/v1/collect",
            "state_path": str(tmp / "telemetry_state.json"),
            "audit_log_path": str(tmp / "telemetry_audit.log"),
        },
        "notifications": {"enabled": False},
    }

    class _Snap:
        def __init__(self, resolved):
            self.resolved = resolved

    monkeypatch.setattr("omega.cli.load_resolved_config", lambda profile: _Snap(cfg))

    st_args = argparse.Namespace(profile="dev", action="status")
    st = json.loads(_run_telemetry(st_args))
    assert st["status"] == "ok"
    assert "telemetry" in st

    pending_args = argparse.Namespace(profile="dev", action="show-pending")
    pending = json.loads(_run_telemetry(pending_args))
    assert pending["status"] == "ok"
    assert pending["result"]["enabled"] is True

    disable_args = argparse.Namespace(profile="dev", action="disable")
    disabled = json.loads(_run_telemetry(disable_args))
    assert disabled["status"] == "ok"
    assert disabled["result"]["enabled"] is False

