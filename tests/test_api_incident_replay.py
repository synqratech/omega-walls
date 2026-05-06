from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
import uuid

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("cryptography")

from fastapi.testclient import TestClient

from omega.api import server as api_server
from omega.api.incident_export import IncidentApiKeyStore, IncidentExportStore, IncidentRateLimiter
from omega.api.incident_replay import IncidentReplayJobManager, IncidentReplayStore
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


def _cfg(
    tmp_path: Path,
    *,
    incident_export_enabled: bool = True,
    incident_replay_enabled: bool = True,
    retention_days: int = 30,
) -> Dict[str, Any]:
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
                "retention_days": int(retention_days),
                "store": {"sqlite_path": str(tmp_path / "incident_export.db")},
                "auth": {
                    "key_store_path": str(tmp_path / "incident_export_keys.db"),
                    "required_scope": "incidents:read",
                },
                "rate_limit": {"rpm": 1200, "burst": 200},
                "cors": {"allowed_origins": []},
            },
            "incident_replay": {
                "enabled": bool(incident_replay_enabled),
                "contract_version": "1.0.0",
                "download_ttl_hours": 24,
                "job_ttl_hours": 72,
                "max_steps": 50,
                "store": {"sqlite_path": str(tmp_path / "incident_replay.db")},
                "package_storage": {
                    "path": str(tmp_path / "replay_packages"),
                    "encryption_key_env": "OMEGA_REPLAY_ENCRYPTION_KEY",
                },
                "worker": {"max_concurrent_jobs": 2},
                "auth": {
                    "required_scopes": {
                        "read": "incidents:replay:read",
                        "raw": "incidents:replay:raw",
                    }
                },
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


def _runtime(
    tmp_path: Path,
    *,
    incident_export_enabled: bool = True,
    incident_replay_enabled: bool = True,
    retention_days: int = 30,
) -> api_server.ScanRuntime:
    cfg = _cfg(
        tmp_path,
        incident_export_enabled=incident_export_enabled,
        incident_replay_enabled=incident_replay_enabled,
        retention_days=retention_days,
    )
    auth_obj = api_server.ApiAuth.from_cfg((cfg.get("api", {}) or {}).get("auth", {}))
    incident_cfg = api_server.IncidentExportConfig.from_cfg((cfg.get("api", {}) or {}).get("incident_export", {}))
    replay_cfg = api_server.IncidentReplayConfig.from_cfg((cfg.get("api", {}) or {}).get("incident_replay", {}))
    incident_store = IncidentExportStore(sqlite_path=incident_cfg.store_path, retention_days=incident_cfg.retention_days)
    key_store = IncidentApiKeyStore(sqlite_path=incident_cfg.key_store_path)
    limiter = IncidentRateLimiter(rpm=incident_cfg.rate_limit_rpm, burst=incident_cfg.rate_limit_burst)
    replay_store = IncidentReplayStore(sqlite_path=replay_cfg.store_path)
    replay_manager = IncidentReplayJobManager(
        config=replay_cfg,
        replay_store=replay_store,
        incident_store=incident_store,
        incident_retention_days=incident_cfg.retention_days,
    )
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
        incident_export_store=(incident_store if incident_export_enabled else None),
        incident_api_key_store=(key_store if incident_export_enabled else None),
        incident_rate_limiter=(limiter if incident_export_enabled else None),
        incident_replay_cfg=(replay_cfg if incident_replay_enabled else None),
        incident_replay_store=(replay_store if incident_replay_enabled else None),
        incident_replay_manager=(replay_manager if incident_replay_enabled else None),
    )


def _client(monkeypatch: pytest.MonkeyPatch, runtime: api_server.ScanRuntime) -> TestClient:
    monkeypatch.setattr(api_server, "_make_runtime", lambda resolved_config: runtime)
    app = api_server.create_app(resolved_config=runtime.config, profile="dev")
    return TestClient(app)


def _scan_headers() -> Dict[str, str]:
    return {"X-API-Key": "scan-api-key"}


def _case_tmp(name: str) -> Path:
    root = Path("tests/_tmp/incident_replay")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _seed_scan_incident(client: TestClient) -> str:
    resp = client.post(
        "/v1/scan/attachment",
        headers=_scan_headers(),
        json={
            "tenant_id": "tenant-a",
            "request_id": f"req-{uuid.uuid4().hex[:8]}",
            "session_id": "sess-a",
            "actor_id": "agent-a",
            "use_extracted_text": True,
            "extracted_text": "safe text",
            "mime": "text/plain",
            "filename": "sample.txt",
        },
    )
    assert resp.status_code == 200
    return str(resp.json().get("incident_export_id", ""))


def test_replay_generate_status_download_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMEGA_REPLAY_ENCRYPTION_KEY", "hex:" + ("11" * 32))
    runtime = _runtime(_case_tmp("flow"), incident_export_enabled=True, incident_replay_enabled=True)
    key = runtime.incident_api_key_store.create_key(scopes=["incidents:read", "incidents:replay:read"])
    with _client(monkeypatch, runtime) as client:
        incident_id = _seed_scan_incident(client)
        headers = {"X-Omega-API-Key": key["api_key"]}
        gen = client.post(
            f"/v1/incidents/{incident_id}/replay/generate",
            headers=headers,
            json={"include_raw_context": False, "redact_sensitive": True, "max_steps": 20, "format": "json_manifest"},
        )
        assert gen.status_code == 202
        job_id = str(gen.json()["job_id"])
        status_body: Dict[str, Any] = {}
        for _ in range(40):
            st = client.get(f"/v1/replay/jobs/{job_id}", headers=headers)
            assert st.status_code == 200
            status_body = st.json()
            if status_body.get("status") in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert status_body.get("status") == "completed"
        dl_url = str(status_body.get("download_url", ""))
        assert dl_url.startswith("/v1/replay/downloads/")
        dl = client.get(dl_url, headers=headers)
        assert dl.status_code == 200
        manifest = dl.json()
        assert manifest["incident_id"] == incident_id
        assert "steps" in manifest
        dl_again = client.get(dl_url, headers=headers)
        assert dl_again.status_code == 404


def test_replay_raw_scope_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMEGA_REPLAY_ENCRYPTION_KEY", "hex:" + ("22" * 32))
    runtime = _runtime(_case_tmp("raw_scope"), incident_export_enabled=True, incident_replay_enabled=True)
    key_read = runtime.incident_api_key_store.create_key(scopes=["incidents:replay:read"])
    key_raw = runtime.incident_api_key_store.create_key(scopes=["incidents:replay:read", "incidents:replay:raw"])
    with _client(monkeypatch, runtime) as client:
        incident_id = _seed_scan_incident(client)
        deny = client.post(
            f"/v1/incidents/{incident_id}/replay/generate",
            headers={"X-Omega-API-Key": key_read["api_key"]},
            json={"include_raw_context": True, "redact_sensitive": True, "max_steps": 10, "format": "json_manifest"},
        )
        assert deny.status_code == 403
        allow = client.post(
            f"/v1/incidents/{incident_id}/replay/generate",
            headers={"X-Omega-API-Key": key_raw["api_key"]},
            json={"include_raw_context": True, "redact_sensitive": False, "max_steps": 10, "format": "json_manifest"},
        )
        assert allow.status_code == 202


def test_replay_generate_retention_expired_returns_410(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMEGA_REPLAY_ENCRYPTION_KEY", "hex:" + ("33" * 32))
    runtime = _runtime(_case_tmp("retention"), incident_export_enabled=True, incident_replay_enabled=True, retention_days=1)
    key = runtime.incident_api_key_store.create_key(scopes=["incidents:replay:read"])
    runtime.incident_export_store.insert_record(
        {
            "incident_id": "00000000-0000-0000-0000-00000000abcd",
            "human_readable_id": "INC-20240101-ABCD",
            "timestamp": "2020-01-01T00:00:00Z",
            "severity": "low",
            "status": "logged_only",
            "attack_type": "unknown",
            "agent_id": "agent",
            "environment": "staging",
            "risk_score": 0.1,
            "source_type": "tool_output",
            "action_taken": "allow_with_flag",
            "steps_count": 1,
            "summary": "old incident",
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
        resp = client.post(
            "/v1/incidents/00000000-0000-0000-0000-00000000abcd/replay/generate",
            headers={"X-Omega-API-Key": key["api_key"]},
            json={"include_raw_context": False, "redact_sensitive": True, "max_steps": 10, "format": "json_manifest"},
        )
        assert resp.status_code == 410
        assert resp.json()["detail"] == "incident_out_of_retention"


def test_replay_feature_flag_disabled_keeps_legacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMEGA_REPLAY_ENCRYPTION_KEY", "hex:" + ("44" * 32))
    runtime = _runtime(_case_tmp("disabled"), incident_export_enabled=True, incident_replay_enabled=False)
    key = runtime.incident_api_key_store.create_key(scopes=["incidents:replay:read"])
    with _client(monkeypatch, runtime) as client:
        incident_id = _seed_scan_incident(client)
        resp = client.post(
            f"/v1/incidents/{incident_id}/replay/generate",
            headers={"X-Omega-API-Key": key["api_key"]},
            json={"include_raw_context": False, "redact_sensitive": True, "max_steps": 10, "format": "json_manifest"},
        )
        assert resp.status_code == 404
