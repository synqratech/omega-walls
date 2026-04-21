from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omega.api import server as api_server
from omega.interfaces.contracts_v1 import OffAction, OffDecision, OmegaOffReasons, OmegaStepResult, ProjectionEvidence, ProjectionResult


def _cfg(
    *,
    mode: str = "stateless",
    allow_request_override: bool = True,
    require_hmac: bool = False,
    require_https: bool = False,
) -> Dict[str, Any]:
    return {
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
                "mode": mode,
                "allow_request_override": bool(allow_request_override),
                "session_store": {
                    "backend": "sqlite",
                    "sqlite_path": "artifacts/state/test_api_session_runtime.db",
                    "session_ttl_sec": 86_400,
                    "request_cache_ttl_sec": 86_400,
                },
            },
            "security": {"transport_mode": "disabled", "require_https": bool(require_https)},
            "auth": {
                "api_keys": ["test-api-key"],
                "require_hmac": bool(require_hmac),
                "hmac_secret_env": "OMEGA_TEST_HMAC_SECRET",
                "hmac_headers": {
                    "signature": "X-Signature",
                    "timestamp": "X-Timestamp",
                    "nonce": "X-Nonce",
                },
                "max_clock_skew_sec": 300,
                "replay_nonce_ttl_sec": 600,
                "replay_cache_max_entries": 10_000,
            },
            "limits": {
                "max_file_bytes": 20 * 1024 * 1024,
                "max_extracted_text_chars": 200_000,
                "request_timeout_sec": 15,
            },
            "logging": {"enabled": False, "include_policy_trace": True},
            "debug": {"enable_document_scan_report": True, "max_report_chunks": 200},
            "chunk_pipeline": {"wall_trigger_threshold": 0.12},
            "policy_mapper": {
                "block_score_threshold": 0.72,
                "quarantine_score_threshold": 0.45,
                "quarantine_worst_threshold": 0.38,
                "quarantine_synergy_threshold": 0.20,
                "exfil_block_wall_threshold": 0.60,
                "confidence_block_threshold": 0.55,
            },
            "attestation": {
                "enabled": False,
                "format": "jws",
                "alg": "RS256",
                "kid": "test-kid",
                "private_key_pem_env": "OMEGA_TEST_ATTESTATION_PRIVATE_KEY",
                "exp_sec": 300,
            },
        },
        "off_policy": {
            "version": "v1",
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
        "retriever": {
            "sqlite_fts": {
                "attachments": {
                    "enabled": True,
                    "max_file_bytes": 20 * 1024 * 1024,
                    "max_extracted_chars": 200_000,
                    "max_chunk_chars": 2000,
                    "chunk_overlap": 200,
                    "zip": {"enabled": False},
                }
            }
        },
    }


class _ProjectorStub:
    def project(self, item: Any) -> ProjectionResult:
        return ProjectionResult(
            doc_id=item.doc_id,
            v=np.zeros(4, dtype=float),
            evidence=ProjectionEvidence(polarity=[0, 0, 0, 0], debug_scores_raw=[0.0, 0.0, 0.0, 0.0], matches={}),
        )


class _StatefulCoreStub:
    def __init__(self) -> None:
        self.calls = 0
        self.params = SimpleNamespace(off_Sigma=1.0)

    def step(self, state: Any, items: list[Any], projections: list[Any]) -> OmegaStepResult:
        _ = projections
        self.calls += 1
        m_prev = np.asarray(state.m, dtype=float)
        m_next = m_prev + np.array([0.2, 0.0, 0.0, 0.0], dtype=float)
        return OmegaStepResult(
            session_id=str(state.session_id),
            step=int(state.step) + 1,
            v_total=np.zeros(4, dtype=float),
            p=np.array([0.1, 0.0, 0.0, 0.0], dtype=float),
            m_prev=m_prev,
            m_next=m_next,
            off=False,
            reasons=OmegaOffReasons(False, False, False, False),
            top_docs=[items[0].doc_id] if items else [],
            contribs=[],
        )


class _PolicyStub:
    def select_actions(self, step_result: OmegaStepResult, items: list[Any]) -> OffDecision:
        _ = (step_result, items)
        return OffDecision(off=False, severity="L1", actions=[])


class _CrossSessionStub:
    def __init__(self, *, source_quarantine: bool = False, tool_freeze: bool = False) -> None:
        self.source_quarantine = bool(source_quarantine)
        self.tool_freeze = bool(tool_freeze)
        self._actor_last_state: Dict[str, np.ndarray] = {}
        self._actor_last_session: Dict[str, str] = {}
        self._carryover_by_session: Dict[tuple[str, str], bool] = {}

    def hydrate_actor_state(self, actor_id: str, session_id: str) -> Any:
        actor = str(actor_id)
        session = str(session_id)
        prev = self._actor_last_state.get(actor)
        prev_session = self._actor_last_session.get(actor)
        carryover = prev is not None and prev_session is not None and prev_session != session
        self._carryover_by_session[(actor, session)] = bool(carryover)
        return SimpleNamespace(
            carryover_applied=bool(carryover),
            carried_scars_after_decay=np.asarray(prev, dtype=float) if carryover else np.zeros(4, dtype=float),
        )

    def record_step(
        self,
        actor_id: str,
        session_id: str,
        step_result: Any,
        policy_actions: list[Any],
        packet_items: list[Any],
    ) -> None:
        _ = (policy_actions, packet_items)
        self._actor_last_state[str(actor_id)] = np.asarray(step_result.m_next, dtype=float)
        self._actor_last_session[str(actor_id)] = str(session_id)

    def active_actions(self, actor_id: str, session_id: str, step: int) -> list[OffAction]:
        _ = (actor_id, step)
        out: list[OffAction] = []
        if self.source_quarantine:
            out.append(
                OffAction(
                    type="SOURCE_QUARANTINE",
                    target="SOURCE",
                    source_ids=[str(session_id)],
                    horizon_steps=8,
                )
            )
        if self.tool_freeze:
            out.append(
                OffAction(
                    type="TOOL_FREEZE",
                    target="TOOLS",
                    tool_mode="TOOLS_DISABLED",
                    allowlist=[],
                    horizon_steps=5,
                )
            )
        return out

    def snapshot(self, actor_id: str, session_id: str, step: int) -> Dict[str, Any]:
        _ = (session_id, step)
        actor = str(actor_id)
        session = str(session_id)
        return {
            "cross_session": {
                "actor_hash": f"hash-{actor_id}",
                "carryover_applied": bool(self._carryover_by_session.get((actor, session), False)),
            }
        }


def _runtime(
    *,
    tmp_path: Path,
    mode: str = "stateless",
    allow_request_override: bool = True,
    require_hmac: bool = False,
    cross_session: Optional[Any] = None,
) -> tuple[api_server.ScanRuntime, _StatefulCoreStub]:
    cfg = _cfg(mode=mode, allow_request_override=allow_request_override, require_hmac=require_hmac)
    sqlite_path = tmp_path / "api-session-runtime.db"
    cfg["api"]["runtime"]["session_store"]["sqlite_path"] = sqlite_path.as_posix()
    core = _StatefulCoreStub()
    auth_obj = api_server.ApiAuth.from_cfg((cfg.get("api", {}) or {}).get("auth", {}))
    runtime_cfg = api_server.ApiRuntime.from_cfg((cfg.get("api", {}) or {}).get("runtime", {}))
    runtime = api_server.ScanRuntime(
        config=cfg,
        projector=_ProjectorStub(),
        omega_core=core,
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
        runtime_cfg=runtime_cfg,
        session_store=api_server.ApiSessionStore(
            sqlite_path=sqlite_path,
            session_ttl_sec=runtime_cfg.session_ttl_sec,
            request_cache_ttl_sec=runtime_cfg.request_cache_ttl_sec,
        ),
        cross_session=cross_session,
    )
    return runtime, core


def _client(monkeypatch: pytest.MonkeyPatch, runtime: api_server.ScanRuntime) -> TestClient:
    monkeypatch.setattr(api_server, "_make_runtime", lambda resolved_config: runtime)
    app = api_server.create_app(resolved_config=runtime.config, profile="dev")
    return TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-api-key"}


def test_stateful_requires_session_id(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime, _ = _runtime(tmp_path=tmp_path)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r1", "runtime_mode": "stateful", "extracted_text": "safe"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "session_id_required_stateful"


def test_runtime_mode_override_can_be_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime, _ = _runtime(tmp_path=tmp_path, allow_request_override=False)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r1", "runtime_mode": "stateful", "extracted_text": "safe"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["policy_trace"]["runtime_mode"] == "stateless"


def test_stateful_persists_step_and_isolates_sessions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime, core = _runtime(tmp_path=tmp_path)
    client = _client(monkeypatch, runtime)

    a1 = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r1", "runtime_mode": "stateful", "session_id": "s1", "extracted_text": "safe"},
    )
    a2 = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r2", "runtime_mode": "stateful", "session_id": "s1", "extracted_text": "safe"},
    )
    b1 = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r3", "runtime_mode": "stateful", "session_id": "s2", "extracted_text": "safe"},
    )

    assert a1.status_code == 200
    assert a2.status_code == 200
    assert b1.status_code == 200
    assert a1.json()["policy_trace"]["state_step_prev"] == 0
    assert a1.json()["policy_trace"]["state_step_next"] == 1
    assert a2.json()["policy_trace"]["state_step_prev"] == 1
    assert a2.json()["policy_trace"]["state_step_next"] == 2
    assert b1.json()["policy_trace"]["state_step_prev"] == 0
    assert b1.json()["policy_trace"]["state_step_next"] == 1
    assert core.calls == 3


def test_stateful_retry_returns_cached_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime, core = _runtime(tmp_path=tmp_path)
    client = _client(monkeypatch, runtime)
    payload = {
        "tenant_id": "t",
        "request_id": "r-cached",
        "runtime_mode": "stateful",
        "session_id": "s-cache",
        "extracted_text": "safe",
    }
    first = client.post("/v1/scan/attachment", headers=_auth_headers(), json=payload)
    second = client.post("/v1/scan/attachment", headers=_auth_headers(), json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == second.json()
    assert core.calls == 1


def test_stateful_applies_active_cross_session_effects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cross = _CrossSessionStub(source_quarantine=True, tool_freeze=True)
    runtime, _ = _runtime(tmp_path=tmp_path, cross_session=cross)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r1", "runtime_mode": "stateful", "session_id": "s1", "extracted_text": "safe"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "quarantine"
    assert "source_quarantine_active" in body["reasons"]
    assert "tool_freeze_active" in body["reasons"]
    assert body["policy_trace"]["runtime_mode"] == "stateful"
    assert body["policy_trace"]["cross_session"]["active_action_types"] == ["SOURCE_QUARANTINE", "TOOL_FREEZE"]


def test_stateful_applies_cross_session_carryover(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    cross = _CrossSessionStub()
    runtime, _ = _runtime(tmp_path=tmp_path, cross_session=cross)
    client = _client(monkeypatch, runtime)

    first = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={
            "tenant_id": "t",
            "request_id": "r1",
            "runtime_mode": "stateful",
            "session_id": "s-a",
            "actor_id": "actor-1",
            "extracted_text": "safe",
        },
    )
    second = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={
            "tenant_id": "t",
            "request_id": "r2",
            "runtime_mode": "stateful",
            "session_id": "s-b",
            "actor_id": "actor-1",
            "extracted_text": "safe",
        },
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["policy_trace"]["cross_session"]["carryover_applied"] is False
    assert second.json()["policy_trace"]["cross_session"]["carryover_applied"] is True
    assert pytest.approx(first.json()["policy_trace"]["sum_m_next"], rel=1e-6) == 0.2
    assert pytest.approx(second.json()["policy_trace"]["sum_m_next"], rel=1e-6) == 0.4


def test_session_reset_endpoint_is_idempotent_and_clears_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime, _ = _runtime(tmp_path=tmp_path)
    client = _client(monkeypatch, runtime)

    first = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r1", "runtime_mode": "stateful", "session_id": "s-reset", "extracted_text": "safe"},
    )
    assert first.status_code == 200
    assert first.json()["policy_trace"]["state_step_prev"] == 0

    reset1 = client.post(
        "/v1/session/reset",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "rs1", "session_id": "s-reset"},
    )
    reset2 = client.post(
        "/v1/session/reset",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "rs2", "session_id": "s-reset"},
    )
    assert reset1.status_code == 200
    assert reset2.status_code == 200
    assert reset1.json()["reset"] is True
    assert reset1.json()["existed"] is True
    assert reset2.json()["reset"] is True
    assert reset2.json()["existed"] is False

    after = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "request_id": "r2", "runtime_mode": "stateful", "session_id": "s-reset", "extracted_text": "safe"},
    )
    assert after.status_code == 200
    assert after.json()["policy_trace"]["state_step_prev"] == 0


def _sign(
    *,
    method: str,
    path: str,
    body_bytes: bytes,
    tenant_id: str,
    request_id: str,
    timestamp: str,
    nonce: str,
    secret: str,
) -> str:
    canonical = api_server._canonical_request_string(
        method=method,
        path=path,
        body_sha256_hex=hashlib.sha256(body_bytes).hexdigest(),
        tenant_id=tenant_id,
        request_id=request_id,
        timestamp=timestamp,
        nonce=nonce,
    )
    return base64.urlsafe_b64encode(
        hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).digest()
    ).decode("ascii").rstrip("=")


def _post_reset_signed(
    *,
    client: TestClient,
    payload: Dict[str, Any],
    secret: str,
    nonce: str,
    ts: Optional[int] = None,
    signature_override: Optional[str] = None,
) -> Any:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ts_i = int(time.time()) if ts is None else int(ts)
    signature = signature_override or _sign(
        method="POST",
        path="/v1/session/reset",
        body_bytes=body,
        tenant_id=str(payload.get("tenant_id", "")),
        request_id=str(payload.get("request_id", "")),
        timestamp=str(ts_i),
        nonce=nonce,
        secret=secret,
    )
    headers = {
        "X-API-Key": "test-api-key",
        "Content-Type": "application/json",
        "X-Signature": signature,
        "X-Timestamp": str(ts_i),
        "X-Nonce": nonce,
    }
    return client.post("/v1/session/reset", headers=headers, content=body)


def test_session_reset_hmac_and_replay(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runtime, _ = _runtime(tmp_path=tmp_path, require_hmac=True)
    client = _client(monkeypatch, runtime)
    secret = "test-secret"
    monkeypatch.setenv("OMEGA_TEST_HMAC_SECRET", secret)
    payload = {"tenant_id": "t", "request_id": "rr1", "session_id": "s1"}

    ok = _post_reset_signed(client=client, payload=payload, secret=secret, nonce="n-1")
    assert ok.status_code == 200
    assert ok.json()["reset"] is True

    bad = _post_reset_signed(client=client, payload=payload, secret=secret, nonce="n-2", signature_override="bad")
    assert bad.status_code == 401
    assert bad.json()["detail"] == "invalid_signature"

    stale = _post_reset_signed(
        client=client,
        payload={"tenant_id": "t", "request_id": "rr2", "session_id": "s1"},
        secret=secret,
        nonce="n-3",
        ts=int(time.time()) - 1000,
    )
    assert stale.status_code == 401
    assert stale.json()["detail"] == "stale_timestamp"

    replay_payload = {"tenant_id": "t", "request_id": "rr3", "session_id": "s1"}
    first = _post_reset_signed(client=client, payload=replay_payload, secret=secret, nonce="n-same")
    second = _post_reset_signed(client=client, payload=replay_payload, secret=secret, nonce="n-same")
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "replay_detected"
