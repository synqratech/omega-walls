from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

from omega.api.incident_export import IncidentApiKeyStore, build_incident_record_from_scan
from omega.cli import _manifest_to_replay_input, _run_keys, _run_replay


def _case_tmp(name: str) -> Path:
    root = Path("tests/_tmp/incident_export")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_incident_record_redacts_fragments() -> None:
    payload = {
        "request_id": "req-1",
        "risk_score": 0.8,
        "control_outcome": "SOFT_BLOCK",
        "policy_trace": {
            "step": 2,
            "severity": "L3",
            "walls_triggered": ["secret_exfiltration"],
            "action_types": ["SOFT_BLOCK"],
        },
        "monitor": {
            "fragments": [
                {
                    "excerpt_redacted": "email john@example.com token=abc12345678901234567890123",
                    "excerpt_sha256": "hash-1",
                    "contribution": 0.6,
                }
            ]
        },
    }
    parsed = {
        "tenant_id": "tenant-a",
        "session_id": "sess-a",
        "actor_id": "agent-a",
        "mime": "application/json",
    }
    record = build_incident_record_from_scan(payload=payload, parsed=parsed, environment="prod", runtime_mode="stateful")
    chain = list(record["chain_of_events"])
    assert chain
    assert "@example.com" not in chain[0]["snippet_redacted"]
    assert "abc12345678901234567890123" not in chain[0]["snippet_redacted"]


def test_key_store_create_rotate_revoke() -> None:
    tmp_path = _case_tmp("keys")
    store = IncidentApiKeyStore(sqlite_path=tmp_path / "keys.db")
    created = store.create_key(scopes=["incidents:read"])
    assert created["api_key"].startswith("owx_")
    resolved = store.resolve_key(provided_key=created["api_key"])
    assert resolved is not None

    rotated = store.rotate(key_id=created["key_id"], scopes=["incidents:read"])
    assert rotated is not None
    assert store.resolve_key(provided_key=created["api_key"]) is None
    assert store.resolve_key(provided_key=rotated["api_key"]) is not None

    assert store.revoke(key_id=rotated["key_id"]) is True
    assert store.resolve_key(provided_key=rotated["api_key"]) is None


def test_cli_keys_lifecycle(monkeypatch) -> None:
    tmp_path = _case_tmp("cli")
    profile_cfg = {
        "api": {
            "incident_export": {
                "auth": {
                    "key_store_path": str(tmp_path / "cli_keys.db"),
                    "required_scope": "incidents:read",
                }
            }
        }
    }

    class _Snap:
        def __init__(self, resolved):
            self.resolved = resolved

    monkeypatch.setattr("omega.cli.load_resolved_config", lambda profile: _Snap(profile_cfg))

    args_create = argparse.Namespace(profile="dev", action="create", key_id=None, scopes="incidents:read")
    out_create = json.loads(_run_keys(args_create))
    assert out_create["status"] == "ok"
    key_id = str((out_create.get("result", {}) or {}).get("key_id", ""))
    assert key_id

    args_list = argparse.Namespace(profile="dev", action="list", key_id=None, scopes="incidents:read")
    out_list = json.loads(_run_keys(args_list))
    assert any(str(row.get("key_id")) == key_id for row in out_list["keys"])

    args_revoke = argparse.Namespace(profile="dev", action="revoke", key_id=key_id, scopes="incidents:read")
    out_revoke = json.loads(_run_keys(args_revoke))
    assert out_revoke["status"] == "ok"


def test_manifest_to_replay_input_minimal() -> None:
    manifest = {
        "incident_id": "inc-1",
        "replay_id": "rpl-1",
        "steps": [
            {
                "step_index": 1,
                "input": {
                    "type": "pdf",
                    "source_hash": "h1",
                    "content_redacted": "step one text",
                    "trust_level": "untrusted",
                },
                "provenance": {"origin_session_id": "sess-1"},
            }
        ],
    }
    out = _manifest_to_replay_input(manifest)
    assert out["event"] == "omega_replay_input_v1"
    assert out["turns"][0]["session_id"] == "sess-1"
    assert out["turns"][0]["packet_items"][0]["source_id"] == "h1"


def test_run_replay_adapter_invokes_backend(monkeypatch) -> None:
    tmp_path = _case_tmp("replay_cli")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "incident_id": "inc-2",
                "replay_id": "rpl-2",
                "steps": [{"step_index": 1, "input": {"content_redacted": "hello"}}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    class _Proc:
        returncode = 0
        stdout = '{"ok": true}'
        stderr = ""

    monkeypatch.setattr("omega.cli.subprocess.run", lambda *args, **kwargs: _Proc())
    args = SimpleNamespace(
        action="run",
        manifest=str(manifest_path),
        sandbox=True,
        strict=False,
        output=None,
        profile="dev",
    )
    out = json.loads(_run_replay(args))
    assert out["status"] == "ok"
    replay_input = Path(str(out["replay_input"]))
    assert replay_input.exists()
