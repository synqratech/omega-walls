"""Runtime artifact normalization and shadow integrity assessment."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from omega.interfaces.contracts_v1 import ContentItem
from omega.runtime.artifacts import ArtifactAssessment, RuntimeArtifact


def runtime_integrity_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("runtime_integrity", {}) if isinstance(config, Mapping) else {}
    return dict(raw) if isinstance(raw, Mapping) else {}


def runtime_integrity_enabled(config: Mapping[str, Any]) -> bool:
    return bool(runtime_integrity_config(config).get("enabled", False))


def classify_artifact_kind(item: ContentItem) -> str:
    meta = dict(item.meta or {})
    origin = str(item.origin or meta.get("origin", "") or "").strip().lower()
    source_type = str(item.source_type or "").strip().lower()
    if origin in {"tool_output", "agent_message"} or source_type == "tool_output":
        return "tool_output"
    if origin in {"memory", "memory_read", "memory_write"} or source_type == "memory":
        return "memory_record"
    if bool(meta.get("semantic_image")) or str(meta.get("attachment_modality", "")).strip().lower() in {
        "image_semantic",
        "ocr",
    }:
        return "visual_attachment"
    if bool(meta.get("command_plan")) or source_type in {"command", "command_plan"}:
        return "command_plan"
    if bool(meta.get("task_file")) or source_type in {"task_file", "repo_file"}:
        return "task_file"
    if source_type in {"skill", "plugin", "connector", "extension", "mcp"}:
        return "skill_package"
    return "context_text"


def build_runtime_artifact(
    item: ContentItem,
    *,
    trusted_control_excluded: bool = False,
    operation_metadata: Optional[Mapping[str, Any]] = None,
) -> RuntimeArtifact:
    meta = dict(item.meta or {})
    artifact_id = str(item.artifact_id or meta.get("artifact_id", "") or "").strip()
    if not artifact_id:
        artifact_id = f"runtime-art-{str(item.doc_id)}"
    content_hash = str(item.content_hash or meta.get("content_hash", "") or "").strip()
    if not content_hash:
        content_hash = f"missing:{artifact_id}"
    derived_from = item.derived_from or meta.get("derived_from", []) or []
    if isinstance(derived_from, str):
        derived_from = [derived_from]
    return RuntimeArtifact(
        artifact_id=artifact_id,
        kind=classify_artifact_kind(item),
        trust_state=str(item.trust or "untrusted"),
        origin=str(item.origin or meta.get("origin", "") or "unknown"),
        source_id=str(item.source_id or ""),
        source_type=str(item.source_type or ""),
        content_hash=content_hash,
        text=str(item.text or ""),
        derived_from=[str(x) for x in list(derived_from or []) if str(x).strip()],
        metadata=meta,
        operation_metadata=dict(operation_metadata or {}),
        trusted_control_excluded=bool(trusted_control_excluded),
    )


def assess_runtime_artifact(
    artifact: RuntimeArtifact,
    *,
    effect_shadow: Optional[Mapping[str, Any]] = None,
) -> ArtifactAssessment:
    integrity_signals = []
    hard_invariant_hits = []
    projection_status = "excluded" if artifact.trusted_control_excluded else "projected"
    effect_signal_status = "none"
    if artifact.trusted_control_excluded:
        integrity_signals.append("trusted_control_excluded")
    if artifact.kind in {"memory_record", "tool_output"} and artifact.trust_state in {
        "trusted",
        "trusted_user",
        "trusted_control",
    }:
        integrity_signals.append("internal_artifact_not_auto_trusted")
    if artifact.kind == "task_file" and artifact.trust_state in {
        "trusted",
        "trusted_control",
    }:
        integrity_signals.append("workspace_artifact_not_auto_trusted")
    if artifact.kind == "visual_attachment":
        integrity_signals.append("visual_provenance_present")
    if artifact.kind == "skill_package":
        integrity_signals.append("skill_boundary_artifact")
    if artifact.kind == "command_plan":
        integrity_signals.append("command_boundary_artifact")
    if bool(artifact.metadata.get("ocr_derived")):
        integrity_signals.append("ocr_derived_text")
    if artifact.metadata.get("integrity_reentry_scanned") is True:
        integrity_signals.append("tool_output_reentry_scanned")

    effect_candidate = None
    if isinstance(effect_shadow, Mapping):
        effect_candidate = effect_shadow.get("effect_wall_candidate")
        effect_signal_status = str(
            effect_shadow.get("effect_policy_gate_status")
            or effect_shadow.get("effect_forecast_status")
            or "none"
        )
        if isinstance(effect_candidate, Mapping):
            integrity_signals.append("packet_level_effect_signal_present")
        named_skill = effect_shadow.get("named_skill_invocation")
        if isinstance(named_skill, Mapping) and bool(named_skill.get("detected", False)):
            integrity_signals.append("named_skill_invocation_detected")
        skill_provenance = effect_shadow.get("skill_provenance_assessment")
        if isinstance(skill_provenance, Mapping):
            status = str(skill_provenance.get("status", "")).strip().lower()
            if status:
                integrity_signals.append(f"skill_provenance_{status}")
        skillbox_verification = effect_shadow.get("skillbox_verification")
        if isinstance(skillbox_verification, Mapping):
            status = str(skillbox_verification.get("verification_status", "")).strip().lower()
            if status:
                integrity_signals.append(f"skillbox_{status}")

    shadow_verdict = "allow"
    requires_approval = False
    if artifact.kind == "skill_package" and artifact.trust_state not in {
        "trusted_control",
        "trusted_user",
    }:
        shadow_verdict = "review"
        requires_approval = True
    elif artifact.kind == "command_plan" and artifact.trust_state not in {
        "trusted_control",
        "trusted_user",
    }:
        shadow_verdict = "review"
    elif artifact.kind == "memory_record" and artifact.trust_state == "tainted_internal":
        shadow_verdict = "review"

    if bool(artifact.metadata.get("source_quarantined")):
        shadow_verdict = "quarantine"
        hard_invariant_hits.append("quarantined_source_artifact")

    return ArtifactAssessment(
        artifact_id=artifact.artifact_id,
        kind=artifact.kind,
        trust_state=artifact.trust_state,
        projection_status=projection_status,
        effect_signal_status=effect_signal_status,
        integrity_signals=integrity_signals,
        shadow_verdict=shadow_verdict,
        would_block=bool(hard_invariant_hits),
        requires_approval=requires_approval,
        hard_invariant_hits=hard_invariant_hits,
        summary={
            "source_id": artifact.source_id,
            "source_type": artifact.source_type,
            "origin": artifact.origin,
            "text_len": int(len(str(artifact.text or ""))),
            "trusted_control_excluded": bool(artifact.trusted_control_excluded),
        },
    )


def summarize_artifact_assessments(
    artifacts: Sequence[RuntimeArtifact],
    assessments: Sequence[ArtifactAssessment],
) -> Dict[str, Any]:
    kind_counts: Dict[str, int] = {}
    trust_counts: Dict[str, int] = {}
    shadow_counts: Dict[str, int] = {}
    hard_hits = []
    named_skill_invocation_count = 0
    skill_provenance_counts: Dict[str, int] = {}
    skillbox_verification_counts: Dict[str, int] = {}
    for artifact in list(artifacts or []):
        kind_counts[str(artifact.kind)] = int(kind_counts.get(str(artifact.kind), 0)) + 1
        trust_counts[str(artifact.trust_state)] = int(
            trust_counts.get(str(artifact.trust_state), 0)
        ) + 1
    for assessment in list(assessments or []):
        shadow_counts[str(assessment.shadow_verdict)] = int(
            shadow_counts.get(str(assessment.shadow_verdict), 0)
        ) + 1
        hard_hits.extend(str(x) for x in list(assessment.hard_invariant_hits or []))
        if "named_skill_invocation_detected" in set(assessment.integrity_signals):
            named_skill_invocation_count += 1
        for signal in list(assessment.integrity_signals or []):
            if not str(signal).startswith("skill_provenance_"):
                if str(signal).startswith("skillbox_"):
                    status = str(signal).removeprefix("skillbox_")
                    skillbox_verification_counts[status] = int(skillbox_verification_counts.get(status, 0)) + 1
                continue
            status = str(signal).removeprefix("skill_provenance_")
            skill_provenance_counts[status] = int(skill_provenance_counts.get(status, 0)) + 1
    return {
        "artifact_count": int(len(list(artifacts or []))),
        "assessment_count": int(len(list(assessments or []))),
        "kind_counts": kind_counts,
        "trust_counts": trust_counts,
        "shadow_verdict_counts": shadow_counts,
        "hard_invariant_hits": sorted(set(hard_hits)),
        "named_skill_invocation_count": int(named_skill_invocation_count),
        "skill_provenance_counts": dict(sorted(skill_provenance_counts.items())),
        "skillbox_verification_counts": dict(sorted(skillbox_verification_counts.items())),
    }
