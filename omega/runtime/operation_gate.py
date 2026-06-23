"""Shadow-first operation gate around runtime action boundaries."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from omega.runtime.artifacts import ArtifactAssessment, OperationGateDecision, OperationIntent


def _cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("runtime_integrity", {}) if isinstance(config, Mapping) else {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def _skillbox_cfg(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("skillbox", {}) if isinstance(config, Mapping) else {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def evaluate_operation_gate(
    *,
    config: Mapping[str, Any],
    intent: OperationIntent,
    artifact_assessments: Sequence[ArtifactAssessment] | None = None,
) -> OperationGateDecision:
    cfg = _cfg(config)
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return OperationGateDecision(
            status="disabled",
            reason_code="runtime_integrity_disabled",
            shadow_only=True,
            would_enforce=False,
            requires_approval=False,
            hard_invariant_hits=[],
            details={"intent": intent.to_dict()},
        )
    mode = str(cfg.get("mode", "shadow")).strip().lower()
    shadow_only = mode != "enforce"
    hard_cfg = dict(cfg.get("hard_invariants", {}) or {})
    resource_cfg = dict(cfg.get("resource_limits", {}) or {})
    approval_cfg = dict(cfg.get("approval_policy", {}) or {})
    skillbox_cfg = _skillbox_cfg(config)
    skillbox_enforcement_cfg = dict(skillbox_cfg.get("enforcement", {}) or {})
    source_mismatch_enforce = bool(skillbox_enforcement_cfg.get("source_mismatch", False))
    assessments = list(artifact_assessments or [])
    source_shadow_verdicts = {
        str(a.shadow_verdict if isinstance(a, ArtifactAssessment) else a.get("shadow_verdict", "allow"))
        for a in assessments
    }
    source_hard_hits = sorted(
        {
            str(hit)
            for assessment in assessments
            for hit in list(
                assessment.hard_invariant_hits
                if isinstance(assessment, ArtifactAssessment)
                else assessment.get("hard_invariant_hits", [])
                or []
            )
            if str(hit).strip()
        }
    )
    details = {
        "intent": intent.to_dict(),
        "source_shadow_verdicts": sorted(source_shadow_verdicts),
        "source_hard_invariant_hits": list(source_hard_hits),
    }

    if (
        str(intent.operation_type) == "memory_write"
        and bool(hard_cfg.get("quarantined_source_memory_write", True))
        and ("quarantine" in source_shadow_verdicts or "quarantined_source_artifact" in source_hard_hits)
    ):
        return OperationGateDecision(
            status="deny",
            reason_code="integrity_quarantined_source_memory_write",
            shadow_only=shadow_only,
            would_enforce=not shadow_only,
            requires_approval=False,
            hard_invariant_hits=["quarantined_source_memory_write"],
            details=details,
        )

    if (
        str(intent.metadata.get("reentry_required", False)).lower() == "true"
        or bool(intent.metadata.get("reentry_required", False))
    ) and bool(hard_cfg.get("missing_reentry_scan", True)):
        if not bool(intent.metadata.get("reentry_scanned", False)):
            return OperationGateDecision(
                status="deny",
                reason_code="integrity_missing_reentry_scan",
                shadow_only=shadow_only,
                would_enforce=not shadow_only,
                requires_approval=False,
                hard_invariant_hits=["missing_reentry_scan"],
                details=details,
            )

    skillbox_status = str(intent.metadata.get("skillbox_verification_status", "")).strip().lower()
    if skillbox_status == "source_mismatch":
        effective_enforce = bool(source_mismatch_enforce) and not shadow_only
        return OperationGateDecision(
            status="deny",
            reason_code="integrity_skillbox_source_mismatch",
            shadow_only=not effective_enforce,
            would_enforce=effective_enforce,
            requires_approval=False,
            hard_invariant_hits=["skillbox_source_mismatch"],
            details=details,
        )

    if str(intent.operation_type) in {"skill_install", "skill_run"}:
        if skillbox_status in {"hash_mismatch", "tampered"}:
            return OperationGateDecision(
                status="deny",
                reason_code=f"integrity_{skillbox_status}",
                shadow_only=True,
                would_enforce=False,
                requires_approval=False,
                hard_invariant_hits=[f"skillbox_{skillbox_status}"],
                details=details,
            )
        if skillbox_status == "dangerous_capability_unapproved":
            return OperationGateDecision(
                status="require_approval",
                reason_code="integrity_skillbox_dangerous_capability_unapproved",
                shadow_only=True,
                would_enforce=False,
                requires_approval=True,
                hard_invariant_hits=[],
                details=details,
            )
        if skillbox_status in {"unknown", "missing_manifest"}:
            return OperationGateDecision(
                status="review",
                reason_code=f"integrity_skillbox_{skillbox_status}",
                shadow_only=True,
                would_enforce=False,
                requires_approval=True,
                hard_invariant_hits=[],
                details=details,
            )
        if bool(intent.metadata.get("source_mismatch", False)) and bool(
            hard_cfg.get("skill_source_mismatch", True)
        ):
            return OperationGateDecision(
                status="deny",
                reason_code="integrity_skill_source_mismatch",
                shadow_only=shadow_only,
                would_enforce=not shadow_only,
                requires_approval=False,
                hard_invariant_hits=["skill_source_mismatch"],
                details=details,
            )
        if (
            bool(hard_cfg.get("untrusted_skill_install_without_approval", True))
            and str(intent.operation_type) == "skill_install"
            and any(trust not in {"trusted_control", "trusted_user"} for trust in list(intent.source_trust_states or []))
            and not bool(intent.approval_present)
        ):
            return OperationGateDecision(
                status="deny",
                reason_code="integrity_untrusted_skill_install_without_approval",
                shadow_only=shadow_only,
                would_enforce=not shadow_only,
                requires_approval=False,
                hard_invariant_hits=["untrusted_skill_install_without_approval"],
                details=details,
            )

    requires_approval = False
    if bool(approval_cfg.get("privileged_requires_approval", True)) and str(
        intent.capability_class or ""
    ).upper() == "PRIV_ESC":
        requires_approval = True
        if not bool(intent.approval_present):
            return OperationGateDecision(
                status="deny",
                reason_code="integrity_privileged_operation_requires_approval",
                shadow_only=shadow_only,
                would_enforce=not shadow_only,
                requires_approval=True,
                hard_invariant_hits=["privileged_operation_without_approval"],
                details=details,
            )

    resource_heavy = bool(intent.metadata.get("resource_heavy", False)) or str(
        intent.operation_type
    ) == "resource_heavy_action"
    if resource_heavy:
        requires_approval = bool(approval_cfg.get("resource_heavy_requires_approval", True)) or requires_approval
        budget_required = bool(resource_cfg.get("require_budget", True))
        budget_available = bool(intent.metadata.get("budget_available", False))
        if budget_required and not budget_available:
            return OperationGateDecision(
                status="deny",
                reason_code="integrity_resource_budget_required",
                shadow_only=shadow_only,
                would_enforce=not shadow_only,
                requires_approval=requires_approval,
                hard_invariant_hits=["resource_budget_required"],
                details=details,
            )
        if requires_approval and not bool(intent.approval_present):
            return OperationGateDecision(
                status="require_approval",
                reason_code="integrity_resource_heavy_requires_approval",
                shadow_only=shadow_only,
                would_enforce=not shadow_only,
                requires_approval=True,
                hard_invariant_hits=[],
                details=details,
            )

    if any(
        trust not in {"trusted_control", "trusted_user"} for trust in list(intent.source_trust_states or [])
    ) and str(intent.operation_type) in {"command_exec", "network_egress", "tool_call"}:
        return OperationGateDecision(
            status="review",
            reason_code="integrity_untrusted_operation_source",
            shadow_only=True,
            would_enforce=False,
            requires_approval=requires_approval,
            hard_invariant_hits=[],
            details=details,
        )

    return OperationGateDecision(
        status="allow",
        reason_code="integrity_operation_allowed",
        shadow_only=shadow_only,
        would_enforce=False,
        requires_approval=requires_approval,
        hard_invariant_hits=[],
        details=details,
    )
