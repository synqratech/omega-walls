from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

import pytest

from omega.cli import _run_alerts, _run_fallback, _run_orchestrator


def _tmp_dir(name: str) -> Path:
    base = Path("artifacts/state").resolve()
    base.mkdir(parents=True, exist_ok=True)
    out = base / f"orch_{name}_{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def _profile_cfg(sqlite_path: Path) -> dict:
    return {
        "projector": {
            "api_perception": {
                "provider": "openai",
                "model": "gpt-5.4-mini",
                "base_url": "https://api.openai.com/v1",
                "orchestrator": {
                    "enabled": True,
                    "master_key_env": "OMEGA_MASTER_KEY",
                    "store": {"sqlite_path": str(sqlite_path)},
                },
            }
        }
    }


def test_orchestrator_cli_keys_and_status(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = _tmp_dir("orch_cli")
    monkeypatch.setenv("OMEGA_MASTER_KEY", "hex:" + "aa" * 32)
    cfg = _profile_cfg(tmp / "orch.db")

    class _Snap:
        def __init__(self, resolved):
            self.resolved = resolved

    monkeypatch.setattr("omega.cli.load_resolved_config", lambda profile: _Snap(cfg))

    add_args = argparse.Namespace(profile="dev", group="keys", action="add", provider="openai-main", key="sk-1234", slot="primary")
    out_add = json.loads(_run_orchestrator(add_args))
    assert out_add["status"] == "ok"

    list_args = argparse.Namespace(profile="dev", group="keys", action="list", provider="openai-main", key=None, slot="primary")
    out_list = json.loads(_run_orchestrator(list_args))
    assert any(str(x.get("provider_id")) == "openai-main" for x in list(out_list.get("keys", [])))

    st_args = argparse.Namespace(profile="dev", group="status", action=None, provider=None, key=None, slot="primary")
    out_st = json.loads(_run_orchestrator(st_args))
    assert out_st["status"] == "ok"
    assert "orchestrator" in out_st


def test_alerts_and_fallback_cli_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = _tmp_dir("orch_cli_cfg")
    monkeypatch.setenv("OMEGA_MASTER_KEY", "hex:" + "bb" * 32)
    cfg = _profile_cfg(tmp / "orch.db")

    class _Snap:
        def __init__(self, resolved):
            self.resolved = resolved

    monkeypatch.setattr("omega.cli.load_resolved_config", lambda profile: _Snap(cfg))

    a_args = argparse.Namespace(
        profile="dev",
        action="configure",
        webhook="https://example.com/hook",
        types="quota_exhausted,fallback_activated",
        channel="webhook",
        duration="1h",
    )
    out_alert_cfg = json.loads(_run_alerts(a_args))
    assert out_alert_cfg["status"] == "ok"

    s_args = argparse.Namespace(
        profile="dev",
        action="silence",
        webhook=None,
        types="",
        channel="webhook",
        duration="30m",
    )
    out_silence = json.loads(_run_alerts(s_args))
    assert out_silence["status"] == "ok"

    f_mode = argparse.Namespace(profile="dev", action="set-mode", mode="fail_closed", errors=None, window=None)
    out_mode = json.loads(_run_fallback(f_mode))
    assert out_mode["status"] == "ok"

    f_thr = argparse.Namespace(profile="dev", action="set-threshold", mode=None, errors=5, window="90s")
    out_thr = json.loads(_run_fallback(f_thr))
    assert out_thr["status"] == "ok"
