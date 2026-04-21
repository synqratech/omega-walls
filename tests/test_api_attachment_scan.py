from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from omega.api import server as api_server
from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import OffAction, OffDecision, OmegaOffReasons, OmegaStepResult, ProjectionEvidence, ProjectionResult
from omega.rag.attachment_ingestion import AttachmentChunk, AttachmentExtractResult


def _base_config(*, attestation_enabled: bool = False, secret_env: str = "OMEGA_TEST_ATTESTATION_SECRET") -> Dict[str, Any]:
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
            "security": {"transport_mode": "disabled", "require_https": False},
            "auth": {
                "api_keys": ["test-api-key"],
                "require_hmac": False,
                "hmac_secret_env": "OMEGA_TEST_HMAC_SECRET",
                "hmac_headers": {
                    "signature": "X-Signature",
                    "timestamp": "X-Timestamp",
                    "nonce": "X-Nonce",
                },
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
            "debug": {"enable_document_scan_report": True, "max_report_chunks": 200},
            "attestation": {
                "enabled": bool(attestation_enabled),
                "format": "jws",
                "alg": "RS256",
                "kid": "test-kid",
                "private_key_pem_env": secret_env,
                "exp_sec": 300,
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
                    "zip": {
                        "enabled": False,
                        "max_files": 100,
                        "max_depth": 5,
                        "max_total_bytes": 20 * 1024 * 1024,
                        "allow_encrypted": False,
                    },
                }
            }
        },
    }


def _step_result(
    *,
    off: bool,
    p: list[float],
    m_next: list[float],
    top_docs: list[str] | None = None,
    reasons: OmegaOffReasons | None = None,
) -> OmegaStepResult:
    reasons_obj = reasons or OmegaOffReasons(False, False, False, False)
    return OmegaStepResult(
        session_id="api:test",
        step=1,
        v_total=np.zeros(4, dtype=float),
        p=np.array(p, dtype=float),
        m_prev=np.zeros(4, dtype=float),
        m_next=np.array(m_next, dtype=float),
        off=bool(off),
        reasons=reasons_obj,
        top_docs=list(top_docs or []),
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
    def __init__(self, step_result: OmegaStepResult, off_sigma: float = 1.0):
        self._step_result = step_result
        self.params = SimpleNamespace(off_Sigma=float(off_sigma))

    def step(self, state: Any, items: list[Any], projections: list[Any]) -> OmegaStepResult:
        return self._step_result


class _PolicyStub:
    def __init__(self, decision: OffDecision):
        self._decision = decision

    def select_actions(self, step_result: OmegaStepResult, items: list[Any]) -> OffDecision:
        return self._decision


def _runtime(
    *,
    step_result: OmegaStepResult,
    severity: str = "L1",
    actions: list[OffAction] | None = None,
    config: Dict[str, Any] | None = None,
    max_text_chars: int = 200_000,
) -> api_server.ScanRuntime:
    cfg = dict(config or _base_config())
    api_cfg = cfg.get("api", {}) or {}
    auth_cfg = (api_cfg.get("auth", {}) or {}) if isinstance(api_cfg.get("auth", {}), dict) else {}
    auth_obj = api_server.ApiAuth.from_cfg(auth_cfg)
    return api_server.ScanRuntime(
        config=cfg,
        projector=_ProjectorStub(),
        omega_core=_CoreStub(step_result=step_result, off_sigma=1.0),
        off_policy=_PolicyStub(
            OffDecision(
                off=bool(step_result.off),
                severity=str(severity),
                actions=list(actions or []),
            )
        ),
        api_keys=["test-api-key"],
        limits=api_server.ApiLimits(
            max_file_bytes=20 * 1024 * 1024,
            max_extracted_text_chars=int(max_text_chars),
            request_timeout_sec=15,
        ),
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


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test-api-key"}


def test_json_extracted_text_contract(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0, 0, 0, 0], m_next=[0, 0, 0, 0]))
    captured: Dict[str, Any] = {}

    def _scan(*, runtime: Any, parsed: Dict[str, Any], include_document_scan_report: bool = False) -> Dict[str, Any]:
        _ = include_document_scan_report
        captured["parsed"] = parsed
        return {
            "request_id": parsed["request_id"],
            "tenant_id": parsed["tenant_id"],
            "risk_score": 0,
            "verdict": "allow",
            "reasons": [],
            "evidence_id": "abc",
            "policy_trace": {"off": False},
        }

    monkeypatch.setattr(api_server, "_scan_request", _scan)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={
            "tenant_id": "tenant-a",
            "request_id": "req-1",
            "extracted_text": "benign text",
            "filename": "note.txt",
            "mime": "text/plain",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["tenant_id"] == "tenant-a"
    assert body["request_id"] == "req-1"
    assert 0 <= int(body["risk_score"]) <= 100
    assert body["verdict"] in {"allow", "quarantine", "block"}
    parsed = captured["parsed"]
    assert parsed["use_extracted_text"] is True
    assert parsed["input_mode"] == "extracted_text"


def test_config_loader_includes_api_block():
    cfg = load_resolved_config(profile="dev").resolved
    assert "api" in cfg
    assert "limits" in (cfg.get("api", {}) or {})
    assert bool((cfg.get("api", {}) or {}).get("auth", {}).get("require_hmac", False)) is True
    assert str((cfg.get("api", {}) or {}).get("security", {}).get("transport_mode", "")) == "proxy_tls"
    assert "debug" in (cfg.get("api", {}) or {})


def test_pilot_profile_enables_document_scan_report():
    cfg = load_resolved_config(profile="pilot").resolved
    assert bool(((cfg.get("api", {}) or {}).get("debug", {}) or {}).get("enable_document_scan_report", False))


def test_json_file_base64_contract(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0, 0, 0, 0], m_next=[0, 0, 0, 0]))
    captured: Dict[str, Any] = {}

    def _scan(*, runtime: Any, parsed: Dict[str, Any], include_document_scan_report: bool = False) -> Dict[str, Any]:
        _ = include_document_scan_report
        captured["parsed"] = parsed
        return {
            "request_id": parsed["request_id"],
            "tenant_id": parsed["tenant_id"],
            "risk_score": 0,
            "verdict": "allow",
            "reasons": [],
            "evidence_id": "x",
            "policy_trace": {"off": False},
        }

    monkeypatch.setattr(api_server, "_scan_request", _scan)
    client = _client(monkeypatch, runtime)
    payload = base64.b64encode(b"hello").decode("ascii")
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "tenant-a", "request_id": "req-b64", "file_base64": payload, "filename": "x.txt"},
    )
    assert resp.status_code == 200
    parsed = captured["parsed"]
    assert parsed["use_extracted_text"] is False
    assert parsed["input_mode"] == "file_base64"
    assert parsed["file_bytes"] == b"hello"


def test_extracted_text_has_priority_over_file(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0, 0, 0, 0], m_next=[0, 0, 0, 0]))
    captured: Dict[str, Any] = {}

    def _scan(*, runtime: Any, parsed: Dict[str, Any], include_document_scan_report: bool = False) -> Dict[str, Any]:
        _ = include_document_scan_report
        captured["parsed"] = parsed
        return {
            "request_id": parsed["request_id"],
            "tenant_id": parsed["tenant_id"],
            "risk_score": 1,
            "verdict": "allow",
            "reasons": [],
            "evidence_id": "x1",
            "policy_trace": {"off": False},
        }

    monkeypatch.setattr(api_server, "_scan_request", _scan)
    client = _client(monkeypatch, runtime)
    payload = base64.b64encode(b"ignored-file").decode("ascii")
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={
            "tenant_id": "tenant-a",
            "request_id": "req-priority",
            "extracted_text": "take-this-text",
            "file_base64": payload,
            "filename": "x.pdf",
            "mime": "application/pdf",
        },
    )
    assert resp.status_code == 200
    parsed = captured["parsed"]
    assert parsed["use_extracted_text"] is True
    assert parsed["input_mode"] == "extracted_text"


def test_multipart_file_contract(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("multipart")
    runtime = _runtime(step_result=_step_result(off=False, p=[0, 0, 0, 0], m_next=[0, 0, 0, 0]))
    captured: Dict[str, Any] = {}

    def _scan(*, runtime: Any, parsed: Dict[str, Any], include_document_scan_report: bool = False) -> Dict[str, Any]:
        _ = include_document_scan_report
        captured["parsed"] = parsed
        return {
            "request_id": parsed["request_id"],
            "tenant_id": parsed["tenant_id"],
            "risk_score": 0,
            "verdict": "allow",
            "reasons": [],
            "evidence_id": "m1",
            "policy_trace": {"off": False},
        }

    monkeypatch.setattr(api_server, "_scan_request", _scan)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        data={"tenant_id": "tenant-m", "request_id": "req-m", "mime": "text/plain"},
        files={"file": ("note.txt", b"safe text", "text/plain")},
    )
    assert resp.status_code == 200
    parsed = captured["parsed"]
    assert parsed["input_mode"] == "file_multipart"
    assert parsed["use_extracted_text"] is False
    assert parsed["filename"] == "note.txt"


def test_invalid_api_key(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0, 0, 0, 0], m_next=[0, 0, 0, 0]))
    client = _client(monkeypatch, runtime)
    resp = client.post("/v1/scan/attachment", headers={"X-API-Key": "wrong"}, json={"tenant_id": "t", "extracted_text": "x"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "unauthorized"


def test_validation_missing_tenant_or_payload(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0, 0, 0, 0], m_next=[0, 0, 0, 0]))
    client = _client(monkeypatch, runtime)

    resp1 = client.post("/v1/scan/attachment", headers=_auth_headers(), json={"extracted_text": "x"})
    assert resp1.status_code == 400
    assert resp1.json()["detail"] == "tenant_id_required"

    resp2 = client.post("/v1/scan/attachment", headers=_auth_headers(), json={"tenant_id": "t"})
    assert resp2.status_code == 400
    assert resp2.json()["detail"] == "missing_payload"


def test_oversize_extracted_text(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(
        step_result=_step_result(off=False, p=[0, 0, 0, 0], m_next=[0, 0, 0, 0]),
        max_text_chars=5,
    )
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "t", "extracted_text": "123456"},
    )
    assert resp.status_code == 413
    assert resp.json()["detail"] == "extracted_text_too_large"


def test_zip_deferred_runtime_quarantine_via_json_file_base64(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]))
    client = _client(monkeypatch, runtime)
    payload = base64.b64encode(b"PK\x03\x04zip-bytes").decode("ascii")
    resp = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={"tenant_id": "tenant-z", "request_id": "req-zip", "filename": "archive.zip", "file_base64": payload},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["verdict"] == "quarantine"
    assert "zip_deferred_runtime" in body["reasons"]
    assert int(body["risk_score"]) >= 60


def test_decision_mapping_allow_block_quarantine():
    parsed = {"tenant_id": "t", "request_id": "r", "use_extracted_text": True, "extracted_text": "safe"}

    allow_runtime = _runtime(step_result=_step_result(off=False, p=[0.01, 0.0, 0.0, 0.0], m_next=[0.01, 0, 0, 0]), severity="L1")
    allow_out = api_server._scan_request(allow_runtime, parsed)
    assert allow_out["verdict"] == "allow"

    block_runtime = _runtime(
        step_result=_step_result(
            off=True,
            p=[0.0, 0.9, 0.0, 0.0],
            m_next=[0.0, 0.7, 0.0, 0.0],
            reasons=OmegaOffReasons(True, True, False, False),
            top_docs=["r:c000"],
        ),
        severity="L3",
        actions=[OffAction(type="SOFT_BLOCK", target="DOC")],
    )
    block_out = api_server._scan_request(block_runtime, parsed)
    assert block_out["verdict"] == "block"

    quarantine_runtime = _runtime(
        step_result=_step_result(
            off=True,
            p=[0.8, 0.0, 0.0, 0.0],
            m_next=[0.2, 0.0, 0.0, 0.0],
            reasons=OmegaOffReasons(False, True, False, False),
        ),
        severity="L2",
    )
    quarantine_out = api_server._scan_request(quarantine_runtime, parsed)
    assert quarantine_out["verdict"] == "quarantine"


def test_hallucination_guard_lite_warns_on_low_confidence_untrusted_allow_path():
    cfg = _base_config()
    cfg["api"].setdefault("policy_mapper", {})["hallucination_guard_lite"] = {
        "enabled": True,
        "apply_when_source_trust": ["untrusted", "mixed"],
        "low_confidence_lte": 0.35,
        "only_if_intended_allow": True,
    }
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]), config=cfg)
    out = api_server._scan_request(
        runtime,
        {"tenant_id": "tenant-hg", "request_id": "req-hg", "use_extracted_text": True, "extracted_text": "safe text"},
    )
    assert out["verdict"] == "allow"
    assert out["control_outcome"] == "WARN"
    assert "hallucination_guard_lite_low_confidence_untrusted" in out["reasons"]
    constraints = out.get("response_constraints", {})
    assert constraints.get("enabled") is True
    assert constraints.get("disclaimer_required") is True
    assert constraints.get("citation_required") is True
    assert constraints.get("reason_code") == "hallucination_guard_lite_low_confidence_untrusted"
    assert isinstance(constraints.get("citation_candidates"), list) and len(constraints["citation_candidates"]) >= 1
    assert constraints.get("suggested_mode") == "answer_with_uncertainty_and_citations"
    assert out["policy_trace"]["hallucination_guard_lite"]["triggered"] is True
    assert out["monitor"]["hallucination_guard_lite"]["triggered"] is True


def test_hallucination_guard_lite_does_not_trigger_for_trusted_sources():
    cfg = _base_config()
    cfg["source_policy"] = {
        "default_trust": "untrusted",
        "source_type_to_trust": {"other": "trusted"},
        "source_prefix_to_trust": {},
    }
    cfg["api"].setdefault("policy_mapper", {})["hallucination_guard_lite"] = {
        "enabled": True,
        "apply_when_source_trust": ["untrusted", "mixed"],
        "low_confidence_lte": 0.35,
        "only_if_intended_allow": True,
    }
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]), config=cfg)
    out = api_server._scan_request(
        runtime,
        {"tenant_id": "tenant-trusted", "request_id": "req-trusted", "use_extracted_text": True, "extracted_text": "safe text"},
    )
    assert out["control_outcome"] == "ALLOW"
    assert "hallucination_guard_lite_low_confidence_untrusted" not in out["reasons"]
    assert out["response_constraints"]["enabled"] is False
    assert out["policy_trace"]["hallucination_guard_lite"]["triggered"] is False
    assert out["policy_trace"]["hallucination_guard_lite"]["source_risk_band"] == "trusted"


def test_hallucination_guard_lite_does_not_trigger_when_confidence_high(monkeypatch: pytest.MonkeyPatch):
    cfg = _base_config()
    cfg["api"].setdefault("policy_mapper", {})["hallucination_guard_lite"] = {
        "enabled": True,
        "apply_when_source_trust": ["untrusted", "mixed"],
        "low_confidence_lte": 0.35,
        "only_if_intended_allow": True,
    }
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]), config=cfg)

    def _score_chunks_high_conf(*, projector: Any, items: list[Any], walls: list[str], cfg: Dict[str, Any] | None = None):
        _ = (projector, walls, cfg)
        return SimpleNamespace(
            chunk_scores=[],
            projections=[],
            wall_max={str(w): 0.0 for w in runtime.config["omega"]["walls"]},
            worst_chunk_score=0.05,
            pattern_synergy=0.05,
            confidence=0.90,
            doc_score=0.10,
            pair_hits=[],
            top_chunks=[{"doc_id": str(items[0].doc_id), "score_max": 0.1, "active_walls": [], "pattern_signals": [], "rule_ids": []}],
            rule_ids=[],
            triggered_chunk_ids=[],
            reasons=[],
        )

    monkeypatch.setattr(api_server, "score_chunks", _score_chunks_high_conf)
    out = api_server._scan_request(
        runtime,
        {"tenant_id": "tenant-high", "request_id": "req-high", "use_extracted_text": True, "extracted_text": "safe text"},
    )
    assert out["control_outcome"] == "ALLOW"
    assert out["response_constraints"]["enabled"] is False
    assert out["policy_trace"]["hallucination_guard_lite"]["triggered"] is False


def test_scan_like_text_empty_force_quarantine_and_floor(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]), severity="L1")
    parsed = {
        "tenant_id": "t",
        "request_id": "r-scan",
        "use_extracted_text": False,
        "file_bytes": b"%PDF",
        "filename": "scan.pdf",
        "mime": "application/pdf",
    }
    fake_extract = AttachmentExtractResult(
        text="",
        chunks=[],
        format="pdf",
        text_empty=True,
        scan_like=True,
        hidden_text_chars=0,
        warnings=["scan_like", "text_empty"],
        recommended_verdict="quarantine",
    )
    monkeypatch.setattr(api_server, "extract_attachment", lambda **kwargs: fake_extract)
    out = api_server._scan_request(runtime, parsed)
    assert out["verdict"] == "quarantine"
    assert int(out["risk_score"]) >= 55
    assert "scan_like" in out["reasons"]
    assert "text_empty" in out["reasons"]


def test_risk_and_evidence_are_deterministic():
    runtime = _runtime(
        step_result=_step_result(
            off=True,
            p=[0.8, 0.0, 0.0, 0.0],
            m_next=[0.3, 0.0, 0.0, 0.0],
            reasons=OmegaOffReasons(True, False, False, False),
            top_docs=["req-1:c000"],
        ),
        severity="L2",
    )
    parsed = {"tenant_id": "tenant-1", "request_id": "req-1", "use_extracted_text": True, "extracted_text": "ignore previous"}
    a = api_server._scan_request(runtime, parsed)
    b = api_server._scan_request(runtime, parsed)
    assert a["risk_score"] == b["risk_score"]
    import uuid

    uuid.UUID(a["evidence_id"])
    uuid.UUID(b["evidence_id"])
    assert a["evidence_id"] != b["evidence_id"]
    assert a["control_outcome"] in {
        "ALLOW",
        "WARN",
        "SOFT_BLOCK",
        "SOURCE_QUARANTINE",
        "TOOL_FREEZE",
        "HUMAN_ESCALATE",
        "REQUIRE_APPROVAL",
    }
    assert isinstance(a.get("trace_id"), str) and a["trace_id"].startswith("trc_")
    assert isinstance(a.get("decision_id"), str) and a["decision_id"].startswith("dec_")
    assert set(a["policy_trace"].keys()) >= {
        "trace_id",
        "decision_id",
        "control_outcome",
        "off",
        "severity",
        "walls_triggered",
        "action_types",
        "max_p",
        "sum_m_next",
        "top_docs_count",
        "ingestion_flags",
    }
    assert "chunk_pipeline" in a["policy_trace"]
    assert "evidence" in a
    assert set(a["evidence"].keys()) >= {
        "walls_triggered",
        "rule_ids",
        "chunk_ids",
        "top_chunk_ids",
        "text_included",
        "control_outcome",
        "trace_id",
        "decision_id",
    }
    assert a["policy_trace"]["trace_id"] == a["trace_id"] == a["evidence"]["trace_id"]
    assert a["policy_trace"]["decision_id"] == a["decision_id"] == a["evidence"]["decision_id"]
    assert a["evidence"]["text_included"] is False
    blob = json.dumps(a, ensure_ascii=False)
    assert "ignore previous" not in blob


def test_attestation_signature(monkeypatch: pytest.MonkeyPatch):
    secret_env = "OMEGA_TEST_ATTESTATION_SECRET"
    private_key = "-----BEGIN PRIVATE KEY-----\nFAKE\n-----END PRIVATE KEY-----"
    monkeypatch.setenv(secret_env, private_key)
    captured: Dict[str, Any] = {}

    def _fake_build_jws(*, claims: Dict[str, Any], kid: str, private_key_pem: str) -> str:
        captured["claims"] = dict(claims)
        captured["kid"] = kid
        captured["key"] = private_key_pem
        return "header.payload.signature"

    monkeypatch.setattr(api_server, "_build_jws_rs256", _fake_build_jws)
    cfg = _base_config(attestation_enabled=True, secret_env=secret_env)
    runtime = _runtime(
        step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]),
        config=cfg,
    )
    parsed = {"tenant_id": "tenant-a", "request_id": "req-a", "use_extracted_text": True, "extracted_text": "safe"}
    out = api_server._scan_request(runtime, parsed)
    att = out.get("attestation")
    assert isinstance(att, dict)
    assert att["alg"] == "RS256"
    assert att["kid"] == "test-kid"
    assert att["jws"] == "header.payload.signature"
    assert captured["kid"] == "test-kid"
    assert captured["key"] == private_key
    assert captured["claims"]["request_id"] == "req-a"
    assert captured["claims"]["tenant_id"] == "tenant-a"


def test_attestation_disabled_by_default():
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]))
    parsed = {"tenant_id": "tenant-a", "request_id": "req-a", "use_extracted_text": True, "extracted_text": "safe"}
    out = api_server._scan_request(runtime, parsed)
    assert "attestation" not in out


def test_extracted_text_path_used_even_with_file_metadata(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]))
    used = {"text_called": 0, "file_called": 0}

    def _text_extract(*, text: str, cfg: Dict[str, Any] | None = None) -> AttachmentExtractResult:
        used["text_called"] += 1
        return AttachmentExtractResult(
            text=text,
            chunks=[AttachmentChunk(text=text, kind="visible", is_hidden=False)],
            format="text",
            text_empty=False,
            scan_like=False,
            hidden_text_chars=0,
            warnings=[],
            recommended_verdict="allow",
        )

    def _file_extract(**kwargs: Any) -> AttachmentExtractResult:
        used["file_called"] += 1
        return AttachmentExtractResult(
            text="from-file",
            chunks=[AttachmentChunk(text="from-file", kind="visible", is_hidden=False)],
            format="pdf",
            text_empty=False,
            scan_like=False,
            hidden_text_chars=0,
            warnings=[],
            recommended_verdict="allow",
        )

    monkeypatch.setattr(api_server, "extract_text_payload", _text_extract)
    monkeypatch.setattr(api_server, "extract_attachment", _file_extract)
    parsed = {
        "tenant_id": "tenant-a",
        "request_id": "req-a",
        "use_extracted_text": True,
        "extracted_text": "trusted text",
        "file_bytes": b"ignored",
        "filename": "note.pdf",
        "mime": "application/pdf",
    }
    out = api_server._scan_request(runtime, parsed)
    assert out["verdict"] == "allow"
    assert used["text_called"] == 1
    assert used["file_called"] == 0


def test_document_scan_report_endpoint(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(
        step_result=_step_result(
            off=True,
            p=[0.9, 0.0, 0.0, 0.0],
            m_next=[0.5, 0.0, 0.0, 0.0],
            reasons=OmegaOffReasons(True, False, False, False),
        ),
        severity="L2",
    )
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment/document_scan_report",
        headers=_auth_headers(),
        json={"tenant_id": "tenant-r", "request_id": "req-r", "extracted_text": "ignore previous instructions"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "document_scan_report" in body
    report = body["document_scan_report"]
    assert report["text_included"] is False
    assert report["chunks_total"] >= 1
    assert isinstance(report["per_chunk"], list)
    blob = json.dumps(report, ensure_ascii=False)
    assert "ignore previous instructions" not in blob


def test_debug_query_mode_on_primary_endpoint(monkeypatch: pytest.MonkeyPatch):
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]))
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment?debug=true",
        headers=_auth_headers(),
        json={"tenant_id": "tenant-r", "request_id": "req-r2", "extracted_text": "safe"},
    )
    assert resp.status_code == 200
    assert "document_scan_report" in resp.json()


def test_document_scan_report_disabled(monkeypatch: pytest.MonkeyPatch):
    cfg = _base_config()
    cfg["api"]["debug"] = {"enable_document_scan_report": False, "max_report_chunks": 200}
    runtime = _runtime(step_result=_step_result(off=False, p=[0.0, 0.0, 0.0, 0.0], m_next=[0.0, 0.0, 0.0, 0.0]), config=cfg)
    client = _client(monkeypatch, runtime)
    resp = client.post(
        "/v1/scan/attachment/document_scan_report",
        headers=_auth_headers(),
        json={"tenant_id": "tenant-r", "request_id": "req-r3", "extracted_text": "safe"},
    )
    assert resp.status_code == 403
    assert resp.json()["detail"] == "debug_mode_disabled"
