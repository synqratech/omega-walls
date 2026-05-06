from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import uuid

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omega.api import server as api_server
from omega.api.incident_export import IncidentApiKeyStore, IncidentExportStore, IncidentRateLimiter
from omega.interfaces.contracts_v1 import OffDecision, OmegaOffReasons, OmegaStepResult, ProjectionEvidence, ProjectionResult


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


class _PolicyStub:
    def select_actions(self, step_result: OmegaStepResult, items: list[Any]) -> OffDecision:
        _ = (step_result, items)
        return OffDecision(off=False, severity="L1", actions=[])


def _cfg(tmp_path: Path, *, incident_export_enabled: bool = True, burst: int = 10, rpm: int = 60) -> Dict[str, Any]:
    return {
        "runtime": {"guard_mode": "enforce"},
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
            "runtime": {
                "mode": "stateless",
                "allow_request_override": True,
                "session_store": {
                    "backend": "sqlite",
                    "sqlite_path": str(tmp_path / "api_session_runtime.db"),
                    "session_ttl_sec": 86_400,
                    "request_cache_ttl_sec": 86_400,
                },
            },
            "security": {"transport_mode": "disabled", "require_https": False},
            "auth": {
                "api_keys": ["scan-api-key"],
                "require_hmac": False,
                "hmac_secret_env": "OMEGA_TEST_HMAC_SECRET",
                "hmac_headers": {"signature": "X-Signature", "timestamp": "X-Timestamp", "nonce": "X-Nonce"},
                "max_clock_skew_sec": 300,
                "replay_nonce_ttl_sec": 600,
                "replay_cache_max_entries": 10000,
            },
            "limits": {
                "max_file_bytes": 20 * 1024 * 1024,
                "max_extracted_text_chars": 200_000,
                "request_timeout_sec": 15,
            },
            "logging": {"enabled": False, "include_policy_trace": True},
            "debug": {"enable_document_scan_report": False, "max_report_chunks": 200},
            "chunk_pipeline": {"wall_trigger_threshold": 0.12},
            "policy_mapper": {
                "block_score_threshold": 0.72,
                "quarantine_score_threshold": 0.45,
                "quarantine_worst_threshold": 0.38,
                "quarantine_synergy_threshold": 0.20,
                "exfil_block_wall_threshold": 0.60,
                "confidence_block_threshold": 0.55,
            },
            "incident_export": {
                "enabled": bool(incident_export_enabled),
                "contract_version": "1.0",
                "default_environment": "staging",
                "retention_days": 30,
                "store": {"sqlite_path": str(tmp_path / "incident_export.db")},
                "auth": {
                    "key_store_path": str(tmp_path / "incident_export_keys.db"),
                    "required_scope": "incidents:read",
                },
                "rate_limit": {"rpm": int(rpm), "burst": int(burst)},
                "cors": {"allowed_origins": []},
            },
        },
        "off_policy": {
            "version": "v1",
            "policy_version": "v1-default",
            "block": {"target": "DOC"},
            "tool_freeze": {"enabled": True, "horizon_steps": 20, "mode": "TOOLS_DISABLED", "allowlist": []},
            "source_quarantine": {"enabled": True, "strikes_to_quarantine": 2, "duration_steps": 24},
            "cross_session": {"enabled": False},
            "escalate": {"enabled": False},
            "severity": {
                "rules": {
                    "L3_if_walls_any": ["secret_exfiltration"],
                    "L3_if_walls_count_gte": 3,
                    "L2_if_walls_any": ["tool_or_action_abuse"],
                    "default": "L1",
                }
            },
        },
        "retriever": {"sqlite_fts": {"attachments": {"enabled": True}}},
        "monitoring": {"enabled": False},
        "notifications": {"enabled": False},
    }


def _runtime(tmp_path: Path, *, incident_export_enabled: bool = True, burst: int = 10, rpm: int = 60) -> api_server.ScanRuntime:
    cfg = _cfg(tmp_path, incident_export_enabled=incident_export_enabled, burst=burst, rpm=rpm)
    auth_obj = api_server.ApiAuth.from_cfg((cfg.get("api", {}) or {}).get("auth", {}))
    incident_cfg = api_server.IncidentExportConfig.from_cfg((cfg.get("api", {}) or {}).get("incident_export", {}))
    store = IncidentExportStore(sqlite_path=incident_cfg.store_path, retention_days=incident_cfg.retention_days)
    key_store = IncidentApiKeyStore(sqlite_path=incident_cfg.key_store_path)
    limiter = IncidentRateLimiter(rpm=incident_cfg.rate_limit_rpm, burst=incident_cfg.rate_limit_burst)
    return api_server.ScanRuntime(
        config=cfg,
        projector=_ProjectorStub(),
        omega_core=_CoreStub(),
        off_policy=_PolicyStub(),
        api_keys=["scan-api-key"],
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
        runtime_cfg=api_server.ApiRuntime.from_cfg((cfg.get("api", {}) or {}).get("runtime", {})),
        session_store=api_server.ApiSessionStore(
            sqlite_path=str((cfg.get("api", {}) or {}).get("runtime", {}).get("session_store", {}).get("sqlite_path")),
            session_ttl_sec=86_400,
            request_cache_ttl_sec=86_400,
        ),
        cross_session=None,
        notification_dispatcher=None,
        monitor_collector=None,
        incident_export_cfg=incident_cfg,
        incident_export_store=(store if incident_export_enabled else None),
        incident_api_key_store=(key_store if incident_export_enabled else None),
        incident_rate_limiter=(limiter if incident_export_enabled else None),
    )


def _client(monkeypatch: pytest.MonkeyPatch, runtime: api_server.ScanRuntime) -> TestClient:
    monkeypatch.setattr(api_server, "_make_runtime", lambda resolved_config: runtime)
    app = api_server.create_app(resolved_config=runtime.config, profile="dev")
    return TestClient(app)


def _scan_headers() -> Dict[str, str]:
    return {"X-API-Key": "scan-api-key"}


def _case_tmp(name: str) -> Path:
    root = Path("tests/_tmp/incident_export")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _seed_scan_incident(client: TestClient) -> None:
    resp = client.post(
        "/v1/scan/attachment",
        headers=_scan_headers(),
        json={
            "tenant_id": "tenant-a",
            "request_id": "req-1",
            "session_id": "sess-a",
            "actor_id": "agent-a",
            "use_extracted_text": True,
            "extracted_text": "safe text",
            "mime": "text/plain",
            "filename": "sample.txt",
        },
    )
    assert resp.status_code == 200


def test_incident_export_endpoints_and_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(_case_tmp("auth"), incident_export_enabled=True)
    created = runtime.incident_api_key_store.create_key(scopes=["incidents:read"])
    with _client(monkeypatch, runtime) as client:
        _seed_scan_incident(client)

        health = client.get("/v1/health")
        assert health.status_code == 200
        body_h = health.json()
        assert body_h["incident_export"]["enabled"] is True

        unauth = client.get("/v1/incidents")
        assert unauth.status_code == 401
        assert unauth.json()["status"] == 401

        forbidden_key = runtime.incident_api_key_store.create_key(scopes=["other:scope"])
        forbidden = client.get("/v1/incidents", headers={"X-Omega-API-Key": forbidden_key["api_key"]})
        assert forbidden.status_code == 403

        ok = client.get("/v1/incidents", headers={"X-Omega-API-Key": created["api_key"]})
        assert ok.status_code == 200
        data = ok.json()
        assert len(data["data"]) >= 1
        incident_id = data["data"][0]["incident_id"]

        detail = client.get(f"/v1/incidents/{incident_id}", headers={"X-Omega-API-Key": created["api_key"]})
        assert detail.status_code == 200
        detail_body = detail.json()
        assert detail_body["incident_id"] == incident_id
        assert "chain_of_events" in detail_body


def test_incident_export_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(_case_tmp("rate"), incident_export_enabled=True, burst=1, rpm=60)
    created = runtime.incident_api_key_store.create_key(scopes=["incidents:read"])
    runtime.incident_export_store.insert_record(
        {
            "incident_id": "00000000-0000-0000-0000-000000000001",
            "human_readable_id": "INC-20260502-0001",
            "timestamp": "2026-05-02T10:00:00Z",
            "severity": "low",
            "status": "logged_only",
            "attack_type": "unknown",
            "agent_id": "agent",
            "environment": "staging",
            "risk_score": 0.1,
            "source_type": "tool_output",
            "action_taken": "allow_with_flag",
            "steps_count": 1,
            "summary": "test",
            "session_id": "sess",
            "tenant_id": "tenant",
            "chain_of_events": [],
            "policy_triggered": "none",
            "threshold_config": {},
            "resolution": {},
            "provenance_verified": True,
        }
    )
    with _client(monkeypatch, runtime) as client:
        headers = {"X-Omega-API-Key": created["api_key"]}
        first = client.get("/v1/incidents", headers=headers)
        assert first.status_code == 200
        assert "X-RateLimit-Remaining" in first.headers

        second = client.get("/v1/incidents", headers=headers)
        assert second.status_code == 429
        body = second.json()
        assert body["status"] == 429


def test_incident_export_pagination_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(_case_tmp("cursor"), incident_export_enabled=True)
    key = runtime.incident_api_key_store.create_key(scopes=["incidents:read"])
    for idx in range(3):
        runtime.incident_export_store.insert_record(
            {
                "incident_id": f"00000000-0000-0000-0000-00000000000{idx}",
                "human_readable_id": f"INC-20260502-000{idx}",
                "timestamp": f"2026-05-02T10:00:0{idx}Z",
                "severity": "low",
                "status": "logged_only",
                "attack_type": "unknown",
                "agent_id": "agent",
                "environment": "staging",
                "risk_score": 0.1,
                "source_type": "tool_output",
                "action_taken": "allow_with_flag",
                "steps_count": 1,
                "summary": "test",
                "session_id": "sess",
                "tenant_id": "tenant",
                "chain_of_events": [],
                "policy_triggered": "none",
                "threshold_config": {},
                "resolution": {},
                "provenance_verified": True,
            }
        )
    with _client(monkeypatch, runtime) as client:
        headers = {"X-Omega-API-Key": key["api_key"]}
        first = client.get("/v1/incidents?limit=2", headers=headers)
        assert first.status_code == 200
        b1 = first.json()
        assert len(b1["data"]) == 2
        assert b1["pagination"]["has_more"] is True
        c = b1["pagination"]["next_cursor"]
        assert c

        second = client.get(f"/v1/incidents?limit=2&cursor={c}", headers=headers)
        assert second.status_code == 200
        b2 = second.json()
        ids1 = {row["incident_id"] for row in b1["data"]}
        ids2 = {row["incident_id"] for row in b2["data"]}
        assert ids1.isdisjoint(ids2)


def test_incident_export_disabled_keeps_legacy_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _runtime(_case_tmp("disabled"), incident_export_enabled=False)
    with _client(monkeypatch, runtime) as client:
        scan = client.post(
            "/v1/scan/attachment",
            headers=_scan_headers(),
            json={
                "tenant_id": "tenant-a",
                "request_id": "req-1",
                "use_extracted_text": True,
                "extracted_text": "safe text",
                "mime": "text/plain",
            },
        )
        assert scan.status_code == 200
        incidents = client.get("/v1/incidents", headers={"X-Omega-API-Key": "whatever"})
        assert incidents.status_code == 404
        assert incidents.json()["status"] == 404
