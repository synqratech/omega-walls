from __future__ import annotations

from copy import deepcopy

import numpy as np

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import OmegaOffReasons, OmegaStepResult, ToolRequest
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from omega.rag.harness import OmegaRAGHarness
from omega.tools.tool_gateway import ToolGatewayV1
from tests.helpers import mk_item


def _mk_tool_wall_step(step: int, session_id: str = "sess-autonomy") -> OmegaStepResult:
    return OmegaStepResult(
        session_id=session_id,
        step=step,
        v_total=np.array([0.1, 0.1, 0.9, 0.1]),
        p=np.array([0.0, 0.0, 0.92, 0.0]),
        m_prev=np.array([0.0, 0.0, 0.0, 0.0]),
        m_next=np.array([0.0, 0.0, 0.6, 0.0]),
        off=True,
        reasons=OmegaOffReasons(False, True, False, False),
        top_docs=["doc-1"],
        contribs=[],
    )


def test_autonomy_soft_profile_resolves_expected_defaults() -> None:
    cfg = load_resolved_config(profile="autonomy_soft").resolved
    assert cfg.get("profiles", {}).get("env") == "autonomy_soft"
    assert cfg.get("runtime", {}).get("guard_mode") == "enforce"
    assert cfg.get("tools", {}).get("autonomy_soft", {}).get("enabled") is True
    assert cfg.get("off_policy", {}).get("autonomy_soft", {}).get("progressive_freeze", {}).get("enabled") is True


def test_autonomy_soft_prod_requires_approval_only_for_high_risk_classes() -> None:
    cfg = load_resolved_config(profile="autonomy_soft").resolved

    stage_gateway = ToolGatewayV1(cfg)
    stage_decision = stage_gateway.enforce(
        ToolRequest(
            tool_name="network_post",
            args={"url": "https://example.com", "payload": "safe"},
            session_id="s",
            step=1,
        ),
        [],
    )
    assert stage_decision.allowed is True

    prod_cfg = deepcopy(cfg)
    prod_cfg["tools"]["autonomy_soft"]["environment_tag"] = "prod"
    prod_gateway = ToolGatewayV1(prod_cfg)

    pending = prod_gateway.enforce(
        ToolRequest(
            tool_name="network_post",
            args={"url": "https://example.com", "payload": "safe"},
            session_id="s",
            step=1,
        ),
        [],
    )
    assert pending.allowed is False
    assert pending.reason == "HUMAN_APPROVAL_REQUIRED"
    assert pending.approval_required is True

    approved = prod_gateway.enforce(
        ToolRequest(
            tool_name="network_post",
            args={"url": "https://example.com", "payload": "safe", "human_approved": True},
            session_id="s",
            step=1,
        ),
        [],
    )
    assert approved.allowed is True

    write_low_risk = prod_gateway.enforce(
        ToolRequest(
            tool_name="write_file",
            args={"filename": "notes.txt", "content": "ok"},
            session_id="s",
            step=1,
        ),
        [],
    )
    assert write_low_risk.allowed is True


def test_autonomy_soft_backup_precondition_blocks_sensitive_actions() -> None:
    cfg = load_resolved_config(profile="autonomy_soft").resolved
    cfg = deepcopy(cfg)
    cfg["tools"]["autonomy_soft"]["backup_safety"]["immutable_backup_ready"] = False

    gateway = ToolGatewayV1(cfg)
    decision = gateway.enforce(
        ToolRequest(
            tool_name="network_post",
            args={"url": "https://example.com", "payload": "safe"},
            session_id="s",
            step=1,
        ),
        [],
    )
    assert decision.allowed is False
    assert decision.reason == "BACKUP_POLICY_PRECONDITION"


def test_autonomy_soft_progressive_freeze_stages_1_2_3() -> None:
    cfg = load_resolved_config(profile="autonomy_soft").resolved
    policy = OffPolicyV1(cfg)
    items = [mk_item("doc-1", "tool abuse", source_id="web:evil")]

    first = policy.select_actions(_mk_tool_wall_step(step=1), items)
    first_freeze = next(a for a in first.actions if a.type == "TOOL_FREEZE")
    assert first_freeze.freeze_stage == 1
    assert first_freeze.tool_mode == "TOOLS_ALLOWLIST"
    assert first_freeze.horizon_steps == 6

    second = policy.select_actions(_mk_tool_wall_step(step=2), items)
    second_freeze = next(a for a in second.actions if a.type == "TOOL_FREEZE")
    assert second_freeze.freeze_stage == 2
    assert second_freeze.tool_mode == "TOOLS_ALLOWLIST"
    assert second_freeze.horizon_steps == 12

    third = policy.select_actions(_mk_tool_wall_step(step=3), items)
    third_freeze = next(a for a in third.actions if a.type == "TOOL_FREEZE")
    assert third_freeze.freeze_stage == 3
    assert third_freeze.tool_mode == "TOOLS_DISABLED"
    assert third_freeze.horizon_steps == 20
    assert third_freeze.escalation_required is True
    assert any(a.type == "HUMAN_ESCALATE" for a in third.actions)


def test_autonomy_soft_observability_fields_are_emitted() -> None:
    cfg = load_resolved_config(profile="autonomy_soft").resolved
    harness = OmegaRAGHarness(
        projector=Pi0IntentAwareV2(cfg),
        omega_core=OmegaCoreV1(omega_params_from_config(cfg)),
        off_policy=OffPolicyV1(cfg),
        tool_gateway=ToolGatewayV1(cfg),
        config=cfg,
    )

    out = harness.run_step(
        user_query="q",
        packet_items=[mk_item("doc-safe", "General harmless information", source_id="tests:safe")],
        tool_requests=[ToolRequest(tool_name="summarize", args={"text": "alpha"}, session_id="sess", step=1)],
    )
    event = out["tool_gateway_events"][0]
    assert "capability_class" in event["decision"]
    assert "risk_level" in event["decision"]
    assert "approval_required" in event["decision"]
    assert "capability_class" in event["capability"]
    assert "risk_level" in event["capability"]
    assert "freeze_stage" in out["enforcement_event"]["freeze"]
    assert "stage_reason" in out["enforcement_event"]["freeze"]
    assert "escalation_required" in out["enforcement_event"]["freeze"]
