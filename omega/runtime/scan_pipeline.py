"""Shared internal scan pipeline phases for API and runtime harness."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from omega.interfaces.contracts_v1 import ContentItem, OffAction, OffDecision, OmegaState
from omega.policy.control_outcome import control_outcome_from_actions


@dataclass(frozen=True)
class ProjectionPhaseState:
    projections: List[Any]
    semantic_failure_policy: str
    semantic_failure_detected: bool
    semantic_failure_status: str
    semantic_policy_branch: str


@dataclass(frozen=True)
class CoreStepPhaseState:
    state: OmegaState
    step_result: Any
    policy_decision: Any


@dataclass(frozen=True)
class ControlOutcomePhaseState:
    walls_triggered: List[str]
    reason_flags: List[str]
    action_types: List[str]
    intended_action_types: List[str]
    semantic_failure_status: str
    semantic_failure_policy: str
    semantic_policy_branch: str


@dataclass(frozen=True)
class EnforcementPhaseState:
    intended_decision: OffDecision
    decision: OffDecision
    enforcement_actions: List[OffAction]
    tools_execution_mode: str
    intended_action_types: List[str]
    action_types: List[str]
    intended_action: str
    actual_action: str


def semantic_failure_policy_from_config(cfg: Mapping[str, Any]) -> str:
    api_cfg = ((cfg.get("projector", {}) or {}).get("api_perception", {}) or {})
    raw = str(api_cfg.get("semantic_failure_policy", "degrade")).strip().lower()
    if raw in {"degrade", "escalate", "fail_closed"}:
        return raw
    return "degrade"


def normalize_action_types(actions: Sequence[OffAction]) -> List[str]:
    return sorted({str(a.type) for a in list(actions or [])})


def compose_effective_actions(
    *,
    policy_actions: Sequence[OffAction],
    cross_active_actions: Sequence[OffAction],
) -> List[OffAction]:
    def _sig(action: OffAction) -> Tuple[str, str, Tuple[str, ...], Tuple[str, ...], str, Tuple[str, ...], int]:
        return (
            str(action.type),
            str(action.target),
            tuple(sorted(str(x) for x in list(action.doc_ids or []))),
            tuple(sorted(str(x) for x in list(action.source_ids or []))),
            str(action.tool_mode or ""),
            tuple(sorted(str(x) for x in list(action.allowlist or []))),
            int(action.horizon_steps or 0),
        )

    seen: Set[Tuple[str, str, Tuple[str, ...], Tuple[str, ...], str, Tuple[str, ...], int]] = set()

    def _append_unique(out: List[OffAction], action: OffAction) -> None:
        key = _sig(action)
        if key in seen:
            return
        seen.add(key)
        out.append(action)

    out: List[OffAction] = []
    # Keep deterministic ordering with policy-first precedence.
    for action in list(policy_actions or []):
        if action.type in {"SOFT_BLOCK", "HUMAN_ESCALATE"}:
            _append_unique(out, action)
    for action in list(policy_actions or []):
        if action.type == "TOOL_FREEZE":
            _append_unique(out, action)
    for action in list(policy_actions or []):
        if action.type == "SOURCE_QUARANTINE":
            _append_unique(out, action)
    for action in list(cross_active_actions or []):
        if action.type == "TOOL_FREEZE":
            _append_unique(out, action)
    for action in list(cross_active_actions or []):
        if action.type == "SOURCE_QUARANTINE":
            _append_unique(out, action)
    for action in list(policy_actions or []):
        if action.type not in {"SOFT_BLOCK", "HUMAN_ESCALATE", "TOOL_FREEZE", "SOURCE_QUARANTINE"}:
            _append_unique(out, action)
    for action in list(cross_active_actions or []):
        if action.type not in {"SOFT_BLOCK", "HUMAN_ESCALATE", "TOOL_FREEZE", "SOURCE_QUARANTINE"}:
            _append_unique(out, action)
    return out


def apply_semantic_failure_policy_to_actions(
    *,
    actions: Sequence[OffAction],
    semantic_phase: ProjectionPhaseState,
    session_id: str,
    step: int,
) -> List[OffAction]:
    out = list(actions or [])
    if bool(semantic_phase.semantic_failure_detected) and str(semantic_phase.semantic_failure_policy) == "fail_closed":
        raise RuntimeError("semantic_failure_fail_closed")
    if bool(semantic_phase.semantic_failure_detected) and str(semantic_phase.semantic_failure_policy) == "escalate":
        if not any(str(a.type) == "HUMAN_ESCALATE" for a in out):
            out.append(
                OffAction(
                    type="HUMAN_ESCALATE",
                    target="AGENT",
                    incident_packet={
                        "reason": "semantic_failure_policy_escalate",
                        "session_id": str(session_id),
                        "step": int(step),
                    },
                )
            )
    return out


def compose_enforcement_phase(
    *,
    policy_decision: OffDecision,
    effective_actions: Sequence[OffAction],
    monitor_enabled: bool,
    enforcement_mode: str,
    tools_execution_mode: str,
) -> EnforcementPhaseState:
    intended_decision = OffDecision(
        off=bool(policy_decision.off),
        severity=str(policy_decision.severity),
        actions=list(effective_actions or []),
        control_outcome=control_outcome_from_actions(list(effective_actions or [])),
    )
    if bool(monitor_enabled):
        decision = OffDecision(
            off=bool(policy_decision.off),
            severity=str(policy_decision.severity),
            actions=[],
            control_outcome="ALLOW",
        )
        enforcement_actions: List[OffAction] = []
        resolved_tools_mode = "DRY_RUN"
    elif str(enforcement_mode).upper() == "ENFORCE":
        decision = OffDecision(
            off=bool(policy_decision.off),
            severity=str(policy_decision.severity),
            actions=list(effective_actions or []),
            control_outcome=control_outcome_from_actions(list(effective_actions or [])),
        )
        enforcement_actions = list(decision.actions)
        resolved_tools_mode = str(tools_execution_mode).upper()
    else:
        decision = OffDecision(
            off=bool(policy_decision.off),
            severity=str(policy_decision.severity),
            actions=[],
            control_outcome="ALLOW",
        )
        enforcement_actions = []
        resolved_tools_mode = str(tools_execution_mode).upper()

    intended_action_types = normalize_action_types(list(intended_decision.actions))
    action_types = normalize_action_types(list(decision.actions))
    return EnforcementPhaseState(
        intended_decision=intended_decision,
        decision=decision,
        enforcement_actions=enforcement_actions,
        tools_execution_mode=resolved_tools_mode,
        intended_action_types=intended_action_types,
        action_types=action_types,
        intended_action=str(intended_decision.control_outcome),
        actual_action=str(decision.control_outcome),
    )


def projection_semantic_failed(projections: Sequence[Any]) -> bool:
    for proj in list(projections or []):
        matches = getattr(getattr(proj, "evidence", None), "matches", {}) or {}
        if not isinstance(matches, Mapping):
            continue
        api_perception = matches.get("api_perception", {})
        if not isinstance(api_perception, Mapping):
            continue
        if bool(api_perception.get("semantic_failed", False)):
            return True
        if str(api_perception.get("semantic_status", "")).strip().lower() == "semantic_failed":
            return True
    return False


def evaluate_projection_phase(
    *,
    cfg: Mapping[str, Any],
    projections: Sequence[Any],
) -> ProjectionPhaseState:
    semantic_failure_policy = semantic_failure_policy_from_config(cfg)
    semantic_failure_detected = projection_semantic_failed(projections)
    semantic_failure_status = "semantic_failed" if semantic_failure_detected else "ok"
    semantic_policy_branch = semantic_failure_policy if semantic_failure_detected else "none"
    return ProjectionPhaseState(
        projections=list(projections),
        semantic_failure_policy=str(semantic_failure_policy),
        semantic_failure_detected=bool(semantic_failure_detected),
        semantic_failure_status=str(semantic_failure_status),
        semantic_policy_branch=str(semantic_policy_branch),
    )


def project_items_phase(
    *,
    projector: Any,
    cfg: Mapping[str, Any],
    items: Sequence[ContentItem],
) -> ProjectionPhaseState:
    projections = [projector.project(item) for item in list(items or [])]
    return evaluate_projection_phase(cfg=cfg, projections=projections)


def dedupe_pressure_items_step_local(
    *,
    items: Sequence[ContentItem],
    current_step: int,
) -> Tuple[List[ContentItem], Dict[str, Any]]:
    def _content_hash(item: ContentItem) -> str:
        if str(item.content_hash or "").strip():
            return str(item.content_hash)
        return hashlib.sha256(str(item.text or "").encode("utf-8", errors="ignore")).hexdigest()

    dedupe_seen: Set[str] = set()
    kept: List[ContentItem] = []
    dropped_rows: List[Dict[str, Any]] = []
    dropped_by_artifact = 0
    dropped_by_content = 0
    step_default = int(current_step)

    for item in list(items or []):
        meta = item.meta or {}
        boundary_step = item.boundary_step
        if boundary_step is None and meta.get("boundary_step") is not None:
            boundary_step = int(meta.get("boundary_step"))
        if boundary_step is None:
            boundary_step = step_default
        boundary_step = int(boundary_step)

        artifact_id = str(item.artifact_id or meta.get("artifact_id", "") or "").strip()
        content_hash = _content_hash(item)
        source_id = str(item.source_id or "")

        dedupe_key = ""
        dedupe_basis = "none"
        if artifact_id:
            dedupe_key = f"{boundary_step}|artifact:{artifact_id}"
            dedupe_basis = "artifact_id"
        else:
            # Preserve source-level signal: same hash from different sources is not deduped.
            dedupe_key = f"{boundary_step}|hash:{content_hash}|source:{source_id}"
            dedupe_basis = "content_hash_source"

        if dedupe_key in dedupe_seen:
            if dedupe_basis == "artifact_id":
                dropped_by_artifact += 1
            else:
                dropped_by_content += 1
            dropped_rows.append(
                {
                    "doc_id": str(item.doc_id),
                    "artifact_id": (artifact_id or None),
                    "content_hash": content_hash,
                    "source_id": source_id,
                    "boundary_step": int(boundary_step),
                    "dedupe_basis": dedupe_basis,
                }
            )
            continue

        dedupe_seen.add(dedupe_key)
        kept.append(item)

    return kept, {
        "enabled": True,
        "input_count": int(len(items or [])),
        "kept_count": int(len(kept)),
        "deduped_count": int(len(dropped_rows)),
        "deduped_by_artifact_id": int(dropped_by_artifact),
        "deduped_by_content_hash_source": int(dropped_by_content),
        "dropped": dropped_rows[:50],
    }


def run_core_step_phase(
    *,
    omega_core: Any,
    off_policy: Any,
    state: OmegaState,
    items: Sequence[ContentItem],
    projections: Sequence[Any],
) -> CoreStepPhaseState:
    step_result = run_omega_step_phase(
        omega_core=omega_core,
        state=state,
        items=items,
        projections=projections,
    )
    policy_decision = off_policy.select_actions(step_result=step_result, items=list(items or []))
    return CoreStepPhaseState(
        state=state,
        step_result=step_result,
        policy_decision=policy_decision,
    )


def run_omega_step_phase(
    *,
    omega_core: Any,
    state: OmegaState,
    items: Sequence[ContentItem],
    projections: Sequence[Any],
) -> Any:
    return omega_core.step(state=state, items=list(items or []), projections=list(projections or []))


def compute_walls_triggered(*, walls: Sequence[str], step_result: Any) -> List[str]:
    out: List[str] = []
    for idx, wall in enumerate(list(walls or [])):
        if float(step_result.p[idx]) > 0.0 or float(step_result.m_next[idx]) > 0.0:
            out.append(str(wall))
    return out


def compute_reason_flags(
    *,
    step_result: Any,
    semantic_phase: Optional[ProjectionPhaseState] = None,
    extra_reason_flags: Optional[Sequence[str]] = None,
) -> List[str]:
    reasons_obj = getattr(step_result, "reasons", None)
    reasons_map = dict(getattr(reasons_obj, "__dict__", {}) or {})
    out = [str(key) for key, value in reasons_map.items() if bool(value)]
    for value in list(extra_reason_flags or []):
        if str(value).strip():
            out.append(str(value))
    if semantic_phase is not None and bool(semantic_phase.semantic_failure_detected):
        out.append("semantic_failed")
        out.append(f"semantic_failure_policy_{semantic_phase.semantic_failure_policy}")
    return sorted({str(x) for x in out if str(x).strip()})


def compose_control_outcome_state(
    *,
    walls: Sequence[str],
    step_result: Any,
    policy_action_types: Sequence[str],
    cross_action_types: Optional[Sequence[str]] = None,
    semantic_phase: Optional[ProjectionPhaseState] = None,
    extra_reason_flags: Optional[Sequence[str]] = None,
) -> ControlOutcomePhaseState:
    action_types = sorted(
        {str(x) for x in list(policy_action_types or []) + list(cross_action_types or []) if str(x).strip()}
    )
    if (
        semantic_phase is not None
        and bool(semantic_phase.semantic_failure_detected)
        and str(semantic_phase.semantic_failure_policy) == "escalate"
        and "HUMAN_ESCALATE" not in set(action_types)
    ):
        action_types = sorted(set(action_types + ["HUMAN_ESCALATE"]))
    reason_flags = compute_reason_flags(
        step_result=step_result,
        semantic_phase=semantic_phase,
        extra_reason_flags=extra_reason_flags,
    )
    walls_triggered = compute_walls_triggered(walls=walls, step_result=step_result)
    semantic_failure_status = str(semantic_phase.semantic_failure_status if semantic_phase is not None else "unknown")
    semantic_failure_policy = str(semantic_phase.semantic_failure_policy if semantic_phase is not None else "none")
    semantic_policy_branch = str(semantic_phase.semantic_policy_branch if semantic_phase is not None else "none")
    return ControlOutcomePhaseState(
        walls_triggered=list(walls_triggered),
        reason_flags=list(reason_flags),
        action_types=list(action_types),
        intended_action_types=list(action_types),
        semantic_failure_status=semantic_failure_status,
        semantic_failure_policy=semantic_failure_policy,
        semantic_policy_branch=semantic_policy_branch,
    )
