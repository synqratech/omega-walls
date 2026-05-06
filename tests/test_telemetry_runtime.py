from __future__ import annotations

import json
from pathlib import Path
import uuid

import jsonschema
import pytest

from omega.telemetry.anonymous import (
    AllowlistSanitizer,
    AnonymousTelemetryService,
    TELEMETRY_BATCH_SCHEMA,
    build_telemetry_event,
)


def _tmp_dir(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid.uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_telemetry_sanitizer_strips_denied_nested_fields() -> None:
    payload = AllowlistSanitizer.sanitize(
        {
            "surface": "runtime",
            "control_outcome": "ALLOW",
            "severity": "L1",
            "attack_types": ["prompt_injection"],
            "policy_triggers": ["reason_spike"],
            "enforcement_actions": ["WARN"],
            "risk_score": 0.22,
            "pattern_parts": ["src:web", "prompt_injection", "outcome:allow"],
            "rule_id": "override_instructions",
            "provenance_type": "web",
            "fp_reported": False,
            "token": "should_not_survive",
            "nested": {"api_key": "secret", "ok": {"password": "x", "safe": 1}},
        }
    )
    as_text = json.dumps(payload, ensure_ascii=False)
    assert "token" not in as_text
    assert "api_key" not in as_text
    assert "password" not in as_text
    assert payload["attack_types"] == ["prompt_injection"]


def test_telemetry_show_pending_schema_valid_and_no_secrets() -> None:
    tmp = _tmp_dir("telemetry_pending")
    cfg = {
        "telemetry": {
            "enabled": True,
            "endpoint": "https://telemetry.omega-walls.io/v1/collect",
            "state_path": str(tmp / "telemetry_state.json"),
            "audit_log_path": str(tmp / "telemetry_audit.log"),
        },
        "notifications": {"enabled": False},
    }
    svc = AnonymousTelemetryService(config=cfg, dispatcher=None, surface="runtime", start_worker=False, emit_startup_notice=False)
    try:
        svc.emit_event(
            build_telemetry_event(
                surface="runtime",
                control_outcome="WARN",
                severity="L2",
                walls_triggered=["override_instructions", "secret_exfiltration"],
                reason_codes=["reason_spike", "text_empty"],
                action_types=["WARN"],
                risk_score=0.48,
                fallback_active=False,
                fallback_level="none",
                accumulation_steps=3,
                provenance_type="pdf",
                module_flags={"monitoring": True, "notifications": True},
            )
        )
        out = svc.show_pending()
        assert out["enabled"] is True
        payload = dict(out["payload"])
        jsonschema.validate(instance=payload, schema=TELEMETRY_BATCH_SCHEMA)
        text = json.dumps(payload, ensure_ascii=False).lower()
        for denied in ("api_key", "token", "password", "hostname", "ip"):
            assert denied not in text
    finally:
        svc.close()


def test_telemetry_retry_drop_after_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    tmp = _tmp_dir("telemetry_retry")
    cfg = {
        "telemetry": {
            "enabled": True,
            "endpoint": "https://telemetry.omega-walls.io/v1/collect",
            "retry_schedule_sec": [1, 1, 1],
            "state_path": str(tmp / "telemetry_state.json"),
            "audit_log_path": str(tmp / "telemetry_audit.log"),
        },
        "notifications": {"enabled": False},
    }
    svc = AnonymousTelemetryService(config=cfg, dispatcher=None, surface="runtime", start_worker=False, emit_startup_notice=False)
    try:
        monkeypatch.setattr("omega.telemetry.anonymous.time.sleep", lambda _: None)

        def _boom(payload):  # type: ignore[no-untyped-def]
            _ = payload
            raise RuntimeError("network_down")

        monkeypatch.setattr(svc._sender, "post", _boom)
        svc.emit_event(
            build_telemetry_event(
                surface="runtime",
                control_outcome="BLOCK",
                severity="L3",
                walls_triggered=["secret_exfiltration"],
                reason_codes=["reason_wall"],
                action_types=["SOFT_BLOCK"],
                risk_score=0.91,
                fallback_active=True,
                fallback_level="rule_only",
                accumulation_steps=7,
                provenance_type="web",
                module_flags={"monitoring": True},
            )
        )
        svc._drain_with_send()
        snap = svc.status_snapshot()
        assert snap["pending_events"] == 0
        assert str((snap.get("last_send_status") or {}).get("status", "")) in {"send_failed_drop", "schema_error_drop", "http_error_drop"}
        assert Path(tmp / "telemetry_audit.log").exists()
    finally:
        svc.close()

