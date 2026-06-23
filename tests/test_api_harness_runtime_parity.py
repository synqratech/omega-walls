from __future__ import annotations

from pathlib import Path

from omega.interfaces.contracts_v1 import ContentItem
from omega.rag.harness import OmegaRAGHarness
from omega.tools.tool_gateway import ToolGatewayV1
from tests.test_api_session_runtime import _cfg, _client, _runtime, _auth_headers, _ProjectorStub, _StatefulCoreStub, _PolicyStub


def _build_stub_harness() -> OmegaRAGHarness:
    cfg = _cfg(mode="stateful", allow_request_override=False, require_hmac=False, require_https=False)
    return OmegaRAGHarness(
        projector=_ProjectorStub(),
        omega_core=_StatefulCoreStub(),
        off_policy=_PolicyStub(),
        tool_gateway=ToolGatewayV1(cfg),
        config=cfg,
    )


def test_api_harness_runtime_parity_shared_fields(tmp_path: Path, monkeypatch):
    runtime, _ = _runtime(tmp_path=tmp_path, mode="stateful")
    client = _client(monkeypatch, runtime)
    response = client.post(
        "/v1/scan/attachment",
        headers=_auth_headers(),
        json={
            "tenant_id": "t",
            "request_id": "r1",
            "runtime_mode": "stateful",
            "session_id": "s1",
            "actor_id": "a1",
            "extracted_text": "safe",
        },
    )
    assert response.status_code == 200
    api_payload = response.json()

    harness = _build_stub_harness()
    h_out = harness.run_step(
        user_query="safe",
        packet_items=[
            ContentItem(
                doc_id="r1:c000",
                source_id="api:t:s1",
                source_type="other",
                trust="untrusted",
                text="safe",
            )
        ],
        actor_id="a1",
    )

    api_trace = api_payload.get("policy_trace", {})
    h_rules = ((h_out.get("monitor", {}) or {}).get("rules", {}) or {})
    h_downstream = ((h_out.get("monitor", {}) or {}).get("downstream", {}) or {})

    assert bool(api_trace.get("off", False)) == bool(h_out["step_result"].off)
    assert str(api_payload.get("control_outcome", "")) == str(h_out.get("control_outcome", ""))
    assert list(api_trace.get("action_types", [])) == list(h_downstream.get("action_types", []))
    assert list(api_trace.get("intended_action_types", [])) == list(h_downstream.get("action_types", []))
    assert list(api_payload.get("reasons", [])) == list(h_rules.get("reason_codes", []))
    assert str(api_trace.get("semantic_failure_status", "")) == str(h_out.get("semantic_failure_status", ""))
    assert str(api_trace.get("semantic_failure_policy", "")) == str(h_out.get("semantic_failure_policy", ""))
    assert str(api_trace.get("semantic_failure_policy_branch", "")) == str(h_out.get("semantic_failure_policy_branch", ""))
    assert isinstance(api_trace.get("artifact_assessment_summary"), dict)
    assert isinstance(h_out.get("artifact_assessment_summary"), dict)
