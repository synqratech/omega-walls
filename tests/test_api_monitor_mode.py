from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omega.api import server as api_server
from omega.interfaces.contracts_v1 import OffAction, OffDecision, OmegaOffReasons, OmegaStepResult, ProjectionEvidence, ProjectionResult
from omega.monitoring.collector import build_monitor_collector_from_config


def _base_config(events_path: Path) -> Dict[str, Any]:
    return {
        "runtime": {"guard_mode": "monitor"},
        "monitoring": {
            "enabled": True,
            "export": {"path": str(events_path), "rotation": "none", "rotation_size_mb": 10, "format": "jsonl"},
        },
        "omega": {
            "walls": [
                "override_instructions",
                "secret_exfiltration",
                "tool_or_action_abuse",
                "policy_evasion",
            ]
        },
        "api": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8080,
            "security": {"transport_mode": "disabled", "require_https": False},
            "auth": {
                "api_keys": ["test-api-key"],
                "require_hmac": False,
                "hmac_secret_env": "OMEGA_TEST_HMAC_SECRET",
                "hmac_headers": {"signature": "X-Signature", "timestamp": "X-Timestamp", "nonce": "X-Nonce"},
                "max_clock_skew_sec": 300,
                "replay_nonce_ttl_sec": 600,
                "replay_cache_max_entries": 1000,
            },
            "limits": {
                "max_file_bytes": 20 * 1024 * 1024,
                "max_extracted_text_chars": 200_000,
                "request_timeout_sec": 15,
            },
            "logging": {"enabled": False, "include_policy_trace": True},
            "debug": {"enable_document_scan_report": False, "max_report_chunks": 200},
        },
        "retriever": {"sqlite_fts": {"attachments": {"enabled": True}}},
    }


def _step_result() -> OmegaStepResult:
    return OmegaStepResult(
        session_id="api:test",
        step=1,
        v_total=np.zeros(4, dtype=float),
        p=np.array([0.0, 0.92, 0.0, 0.0], dtype=float),
        m_prev=np.zeros(4, dtype=float),
        m_next=np.array([0.0, 0.72, 0.0, 0.0], dtype=float),
        off=True,
        reasons=OmegaOffReasons(True, True, False, False),
        top_docs=["req-1:c000"],
        contribs=[],
    )


class _ProjectorStub:
    def project(self, item: Any) -> ProjectionResult:
        return ProjectionResult(
            doc_id=item.doc_id,
            v=np.zeros(4, dtype=float),
            evidence=ProjectionEvidence(polarity=[0, 0, 0, 0], debug_scores_raw=[0.0, 0.0, 0.0, 0.0], matches={}),
        )


class _CoreStub:
    def __init__(self) -> None:
        self.params = SimpleNamespace(off_Sigma=1.0)

    def step(self, state: Any, items: list[Any], projections: list[Any]) -> OmegaStepResult:
        _ = (state, items, projections)
        return _step_result()


class _PolicyStub:
    def select_actions(self, step_result: OmegaStepResult, items: list[Any]) -> OffDecision:
        _ = (step_result, items)
        return OffDecision(
            off=True,
            severity="L3",
            actions=[OffAction(type="SOFT_BLOCK", target="DOC"), OffAction(type="HUMAN_ESCALATE", target="AGENT")],
        )


class _CoreAllowStub:
    def __init__(self) -> None:
        self.params = SimpleNamespace(off_Sigma=1.0)

    def step(self, state: Any, items: list[Any], projections: list[Any]) -> OmegaStepResult:
        _ = (state, items, projections)
        return OmegaStepResult(
            session_id="api:test",
            step=1,
            v_total=np.zeros(4, dtype=float),
            p=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
            m_prev=np.zeros(4, dtype=float),
            m_next=np.zeros(4, dtype=float),
            off=False,
            reasons=OmegaOffReasons(False, False, False, False),
            top_docs=["req-1:c000"],
            contribs=[],
        )


class _PolicyAllowStub:
    def select_actions(self, step_result: OmegaStepResult, items: list[Any]) -> OffDecision:
        _ = (step_result, items)
        return OffDecision(off=False, severity="L1", actions=[])


def _runtime(events_path: Path) -> api_server.ScanRuntime:
    cfg = _base_config(events_path)
    auth_obj = api_server.ApiAuth.from_cfg((cfg.get("api", {}) or {}).get("auth", {}))
    return api_server.ScanRuntime(
        config=cfg,
        projector=_ProjectorStub(),
        omega_core=_CoreStub(),
        off_policy=_PolicyStub(),
        api_keys=["test-api-key"],
        limits=api_server.ApiLimits.from_cfg((cfg.get("api", {}) or {}).get("limits", {})),
        security=api_server.ApiSecurity.from_cfg((cfg.get("api", {}) or {}).get("security", {})),
        auth=auth_obj,
        attestation=api_server.ApiAttestation.from_cfg((cfg.get("api", {}) or {}).get("attestation", {})),
        logging_cfg=api_server.ApiLogging.from_cfg((cfg.get("api", {}) or {}).get("logging", {})),
        debug=api_server.ApiDebug.from_cfg((cfg.get("api", {}) or {}).get("debug", {})),
        replay_cache=api_server.NonceReplayCache(
            ttl_sec=int(auth_obj.replay_nonce_ttl_sec),
            max_entries=int(auth_obj.replay_cache_max_entries),
        ),
        monitor_collector=build_monitor_collector_from_config(config=cfg, force_enable=True),
    )


def _client(monkeypatch: pytest.MonkeyPatch, runtime: api_server.ScanRuntime) -> TestClient:
    monkeypatch.setattr(api_server, "_make_runtime", lambda resolved_config: runtime)
    app = api_server.create_app(resolved_config=runtime.config, profile="dev")
    return TestClient(app)


def test_api_monitor_mode_returns_intended_vs_actual(tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    runtime = _runtime(events_path)
    out = api_server._scan_request(
        runtime,
        {"tenant_id": "tenant-a", "request_id": "req-1", "use_extracted_text": True, "extracted_text": "attack"},
    )
    assert out["control_outcome"] == "ALLOW"
    assert out["monitor"]["enabled"] is True
    assert out["monitor"]["intended_action"] != "ALLOW"
    assert out["monitor"]["actual_action"] == "ALLOW"
    assert events_path.exists()
    rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["mode"] == "monitor"
    assert rows[0]["intended_action"] == out["monitor"]["intended_action"]
    assert rows[0]["actual_action"] == "ALLOW"
    assert isinstance(rows[0].get("fragments", []), list)
    assert isinstance(rows[0].get("downstream", {}), dict)
    assert isinstance(rows[0].get("rules", {}), dict)


def test_api_monitor_health_endpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = _runtime(tmp_path / "monitor_events.jsonl")
    with _client(monkeypatch, runtime) as client:
        resp = client.get("/v1/monitor/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["guard_mode"] == "monitor"
        assert "events_total" in body
        assert body["enabled"] is True


def test_api_monitor_mode_hallucination_guard_lite_sets_intended_warn(tmp_path: Path) -> None:
    events_path = tmp_path / "monitor_events.jsonl"
    runtime = _runtime(events_path)
    runtime.omega_core = _CoreAllowStub()
    runtime.off_policy = _PolicyAllowStub()
    runtime.config.setdefault("api", {}).setdefault("policy_mapper", {})["hallucination_guard_lite"] = {
        "enabled": True,
        "apply_when_source_trust": ["untrusted", "mixed"],
        "low_confidence_lte": 0.35,
        "only_if_intended_allow": True,
    }
    out = api_server._scan_request(
        runtime,
        {"tenant_id": "tenant-a", "request_id": "req-hg-monitor", "use_extracted_text": True, "extracted_text": "safe"},
    )
    assert out["control_outcome"] == "ALLOW"
    assert out["monitor"]["intended_action"] == "WARN"
    assert out["monitor"]["actual_action"] == "ALLOW"
    assert out["monitor"]["response_constraints"]["enabled"] is True
    assert out["monitor"]["hallucination_guard_lite"]["triggered"] is True
