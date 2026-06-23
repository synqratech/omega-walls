from __future__ import annotations

import numpy as np

from omega.interfaces.contracts_v1 import OffAction, OffDecision, OmegaOffReasons, OmegaStepResult
from omega.runtime.scan_pipeline import (
    EnforcementPhaseState,
    ProjectionPhaseState,
    apply_semantic_failure_policy_to_actions,
    compose_effective_actions,
    compose_enforcement_phase,
    compose_control_outcome_state,
)


def _step_result() -> OmegaStepResult:
    return OmegaStepResult(
        session_id="s1",
        step=1,
        v_total=np.zeros(4, dtype=float),
        p=np.array([0.5, 0.0, 0.0, 0.0], dtype=float),
        m_prev=np.zeros(4, dtype=float),
        m_next=np.array([0.5, 0.0, 0.0, 0.0], dtype=float),
        off=True,
        reasons=OmegaOffReasons(True, False, False, False),
        top_docs=["d1"],
        contribs=[],
    )


def test_compose_effective_actions_order_contract() -> None:
    policy = [
        OffAction(type="WARN", target="SESSION"),
        OffAction(type="SOFT_BLOCK", target="DOC", doc_ids=["d1"]),
    ]
    cross = [
        OffAction(type="TOOL_FREEZE", target="TOOLS"),
        OffAction(type="SOURCE_QUARANTINE", target="SOURCE", source_ids=["src:1"]),
    ]
    out = compose_effective_actions(policy_actions=policy, cross_active_actions=cross)
    assert [a.type for a in out] == ["SOFT_BLOCK", "TOOL_FREEZE", "SOURCE_QUARANTINE", "WARN"]


def test_compose_effective_actions_keeps_policy_freeze_and_quarantine() -> None:
    policy = [
        OffAction(type="SOFT_BLOCK", target="DOC", doc_ids=["d1"]),
        OffAction(type="TOOL_FREEZE", target="TOOLS", tool_mode="TOOLS_DISABLED"),
        OffAction(type="SOURCE_QUARANTINE", target="SOURCE", source_ids=["src:policy"]),
    ]
    out = compose_effective_actions(policy_actions=policy, cross_active_actions=[])
    assert [a.type for a in out] == ["SOFT_BLOCK", "TOOL_FREEZE", "SOURCE_QUARANTINE"]


def test_semantic_failure_policy_escalate_injects_human_escalate() -> None:
    phase = ProjectionPhaseState(
        projections=[],
        semantic_failure_policy="escalate",
        semantic_failure_detected=True,
        semantic_failure_status="semantic_failed",
        semantic_policy_branch="escalate",
    )
    out = apply_semantic_failure_policy_to_actions(
        actions=[OffAction(type="WARN", target="SESSION")],
        semantic_phase=phase,
        session_id="s1",
        step=2,
    )
    assert "HUMAN_ESCALATE" in {a.type for a in out}


def test_compose_enforcement_phase_monitor_contract() -> None:
    decision = OffDecision(off=True, severity="L2", actions=[OffAction(type="SOFT_BLOCK", target="DOC")])
    phase: EnforcementPhaseState = compose_enforcement_phase(
        policy_decision=decision,
        effective_actions=list(decision.actions),
        monitor_enabled=True,
        enforcement_mode="ENFORCE",
        tools_execution_mode="ENFORCE",
    )
    assert phase.decision.control_outcome == "ALLOW"
    assert phase.tools_execution_mode == "DRY_RUN"
    assert phase.intended_action_types == ["SOFT_BLOCK"]
    assert phase.action_types == []


def test_compose_enforcement_phase_log_only_reports_allow_actual() -> None:
    decision = OffDecision(off=True, severity="L3", actions=[OffAction(type="HUMAN_ESCALATE", target="AGENT")])
    phase: EnforcementPhaseState = compose_enforcement_phase(
        policy_decision=decision,
        effective_actions=list(decision.actions),
        monitor_enabled=False,
        enforcement_mode="LOG_ONLY",
        tools_execution_mode="ENFORCE",
    )
    assert phase.intended_action == "HUMAN_ESCALATE"
    assert phase.actual_action == "ALLOW"
    assert phase.intended_action_types == ["HUMAN_ESCALATE"]
    assert phase.action_types == []


def test_runtime_orchestration_state_fields_stable() -> None:
    phase = ProjectionPhaseState(
        projections=[],
        semantic_failure_policy="degrade",
        semantic_failure_detected=False,
        semantic_failure_status="ok",
        semantic_policy_branch="none",
    )
    state = compose_control_outcome_state(
        walls=["override_instructions", "secret_exfiltration", "tool_or_action_abuse", "policy_evasion"],
        step_result=_step_result(),
        policy_action_types=["WARN"],
        semantic_phase=phase,
        extra_reason_flags=["ingestion_error"],
    )
    assert state.semantic_failure_status == "ok"
    assert state.semantic_failure_policy == "degrade"
    assert state.semantic_policy_branch == "none"
    assert state.walls_triggered == ["override_instructions"]
    assert "reason_spike" in state.reason_flags
    assert "ingestion_error" in state.reason_flags
