from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omega.api import server as api_server
from omega.interfaces.contracts_v1 import OffDecision, OmegaOffReasons, OmegaStepResult, ProjectionEvidence, ProjectionResult


def _step_result() -> OmegaStepResult:
    return OmegaStepResult(
        session_id="api:test",
        step=1,
        v_total=np.zeros(4, dtype=float),
        p=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        m_prev=np.zeros(4, dtype=float),
        m_next=np.zeros(4, dtype=float),
        off=False,
        reasons=OmegaOffReasons(False, False, False, False),
        top_docs=[],
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
        return _step_result()


class _PolicyStub:
    def select_actions(self, step_result: OmegaStepResult, items: list[Any]) -> OffDecision:
        return OffDecision(off=False, severity="L1", actions=[])


def _cfg(*, require_hmac: bool, require_https: bool, attestation_enabled: bool = False) -> Dict[str, Any]:
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
            "security": {
                "transport_mode": "proxy_tls",
                "require_https": bool(require_https),
            },
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
                "replay_cache_max_entries": 10000,
            },
            "limits": {
                "max_file_bytes": 20 * 1024 * 1024,
                "max_extracted_text_chars": 200_000,
                "request_timeout_sec": 15,
            },
            "logging": {
                "enabled": True,
                "include_policy_trace": True,
            },
            "debug": {
                "enable_document_scan_report": True,
                "max_report_chunks": 200,
            },
            "chunk_pipeline": {
                "wall_trigger_threshold": 0.12,
            },
            "policy_mapper": {
                "block_score_threshold": 0.72,
                "quarantine_score_threshold": 0.45,
                "quarantine_worst_threshold": 0.38,
                "quarantine_synergy_threshold": 0.20,
                "exfil_block_wall_threshold": 0.60,
                "confidence_block_threshold": 0.55,
            },
            "attestation": {
                "enabled": bool(attestation_enabled),
                "format": "jws",
                "alg": "RS256",
                "kid": "test-kid",
                "private_key_pem_env": "OMEGA_TEST_ATTESTATION_PRIVATE_KEY",
                "exp_sec": 300,
            },
        },
        "retriever": {"sqlite_fts": {"attachments": {"enabled": True}}},
    }


def _runtime(cfg: Dict[str, Any]) -> api_server.ScanRuntime:
    api_cfg = cfg.get("api", {}) or {}
    auth_obj = api_server.ApiAuth.from_cfg(api_cfg.get("auth", {}))
    return api_server.ScanRuntime(
        config=cfg,
        projector=_ProjectorStub(),
        omega_core=_CoreStub(),
        off_policy=_PolicyStub(),
        api_keys=["test-api-key"],
        limits=api_server.ApiLimits.from_cfg(api_cfg.get("limits", {})),
        security=api_server.ApiSecurity.from_cfg(api_cfg.get("security", {})),
        auth=auth_obj,
        attestation=api_server.ApiAttestation.from_cfg(api_cfg.get("attestation", {})),
        logging_cfg=api_server.ApiLogging.from_cfg(api_cfg.get("logging", {})),
        debug=api_server.ApiDebug.from_cfg(api_cfg.get("debug", {})),
        replay_cache=api_server.NonceReplayCache(
            ttl_sec=int(auth_obj.replay_nonce_ttl_sec),
            max_entries=int(auth_obj.replay_cache_max_entries),
        ),
    )


def _client(monkeypatch: pytest.MonkeyPatch, runtime: api_server.ScanRuntime) -> TestClient:
    monkeypatch.setattr(api_server, "_make_runtime", lambda resolved_config: runtime)
    app = api_server.create_app(resolved_config=runtime.config, profile="dev")
    return TestClient(app)


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


def _post_signed_json(
    *,
    client: TestClient,
    payload: Dict[str, Any],
    secret: str,
    nonce: str,
    ts: int | None = None,
    include_tls: bool = True,
    signature_override: str | None = None,
) -> Any:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode("utf-8")
    ts_i = int(time.time()) if ts is None else int(ts)
    signature = signature_override or _sign(
        method="POST",
        path="/v1/scan/attachment",
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
    if include_tls:
        headers["X-Forwarded-Proto"] = "https"
    return client.post("/v1/scan/attachment", headers=headers, content=body)


def test_proxy_tls_required(monkeypatch: pytest.MonkeyPatch):
    cfg = _cfg(require_hmac=False, require_https=True)
    runtime = _runtime(cfg)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment",
        headers={"X-API-Key": "test-api-key"},
        json={"tenant_id": "t", "request_id": "r1", "extracted_text": "safe"},
    )
    assert resp.status_code == 400
    assert resp.json()["detail"] == "insecure_transport"


def test_hmac_success_and_failures(monkeypatch: pytest.MonkeyPatch):
    cfg = _cfg(require_hmac=True, require_https=False)
    runtime = _runtime(cfg)
    client = _client(monkeypatch, runtime)
    secret = "test-secret"
    monkeypatch.setenv("OMEGA_TEST_HMAC_SECRET", secret)
    payload = {"tenant_id": "t", "request_id": "r1", "extracted_text": "safe"}

    ok = _post_signed_json(client=client, payload=payload, secret=secret, nonce="n-1")
    assert ok.status_code == 200

    missing_sig = client.post(
        "/v1/scan/attachment",
        headers={"X-API-Key": "test-api-key", "Content-Type": "application/json"},
        content=json.dumps(payload).encode("utf-8"),
    )
    assert missing_sig.status_code == 401
    assert missing_sig.json()["detail"] == "invalid_signature"

    bad_sig = _post_signed_json(client=client, payload=payload, secret=secret, nonce="n-2", signature_override="bad")
    assert bad_sig.status_code == 401
    assert bad_sig.json()["detail"] == "invalid_signature"

    stale = _post_signed_json(
        client=client,
        payload=payload,
        secret=secret,
        nonce="n-3",
        ts=int(time.time()) - 1000,
    )
    assert stale.status_code == 401
    assert stale.json()["detail"] == "stale_timestamp"


def test_replay_nonce_detected(monkeypatch: pytest.MonkeyPatch):
    cfg = _cfg(require_hmac=True, require_https=False)
    runtime = _runtime(cfg)
    client = _client(monkeypatch, runtime)
    secret = "test-secret"
    monkeypatch.setenv("OMEGA_TEST_HMAC_SECRET", secret)
    payload = {"tenant_id": "t", "request_id": "r-replay", "extracted_text": "safe"}
    first = _post_signed_json(client=client, payload=payload, secret=secret, nonce="same-nonce")
    second = _post_signed_json(client=client, payload=payload, secret=secret, nonce="same-nonce")
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"] == "replay_detected"


def test_audit_log_has_no_raw_payload(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture):
    cfg = _cfg(require_hmac=False, require_https=False)
    runtime = _runtime(cfg)
    client = _client(monkeypatch, runtime)
    text = "sensitive body should not appear in logs"
    with caplog.at_level("INFO"):
        resp = client.post(
            "/v1/scan/attachment",
            headers={"X-API-Key": "test-api-key"},
            json={"tenant_id": "t", "request_id": "r-log", "extracted_text": text},
        )
    assert resp.status_code == 200
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "api_scan_audit" in joined
    assert text not in joined
    assert "tenant_id_hash" in joined
    assert "pattern_ids" in joined


def test_jws_attestation_verification(monkeypatch: pytest.MonkeyPatch):
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa

    cfg = _cfg(require_hmac=False, require_https=False, attestation_enabled=True)
    runtime = _runtime(cfg)
    client = _client(monkeypatch, runtime)

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    monkeypatch.setenv("OMEGA_TEST_ATTESTATION_PRIVATE_KEY", private_pem)

    resp = client.post(
        "/v1/scan/attachment",
        headers={"X-API-Key": "test-api-key"},
        json={"tenant_id": "t", "request_id": "r-jws", "extracted_text": "safe"},
    )
    assert resp.status_code == 200
    body = resp.json()
    att = body.get("attestation")
    assert isinstance(att, dict)
    assert att["alg"] == "RS256"
    assert att["kid"] == "test-kid"
    token = str(att["jws"])
    head_b64, payload_b64, sig_b64 = token.split(".")
    signing_input = f"{head_b64}.{payload_b64}".encode("ascii")
    signature = api_server._b64url_decode(sig_b64)
    public_key = private_key.public_key()
    public_key.verify(signature, signing_input, padding.PKCS1v15(), hashes.SHA256())
    claims = json.loads(api_server._b64url_decode(payload_b64).decode("utf-8"))
    assert claims["request_id"] == "r-jws"
    assert claims["tenant_id"] == "t"
    assert "verdict" in claims
    assert "risk_score" in claims
    assert "evidence_id" in claims
    assert "iat" in claims and "exp" in claims


def test_attestation_unavailable_adds_reason(monkeypatch: pytest.MonkeyPatch):
    cfg = _cfg(require_hmac=False, require_https=False, attestation_enabled=True)
    runtime = _runtime(cfg)
    client = _client(monkeypatch, runtime)
    monkeypatch.delenv("OMEGA_TEST_ATTESTATION_PRIVATE_KEY", raising=False)
    resp = client.post(
        "/v1/scan/attachment",
        headers={"X-API-Key": "test-api-key"},
        json={"tenant_id": "t", "request_id": "r-no-key", "extracted_text": "safe"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "attestation" not in body
    assert "attestation_unavailable" in body.get("reasons", [])


def test_document_scan_report_requires_auth(monkeypatch: pytest.MonkeyPatch):
    cfg = _cfg(require_hmac=False, require_https=False)
    runtime = _runtime(cfg)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment/document_scan_report",
        json={"tenant_id": "t", "request_id": "r-report", "extracted_text": "safe"},
    )
    assert resp.status_code == 401
    assert resp.json()["detail"] == "unauthorized"
