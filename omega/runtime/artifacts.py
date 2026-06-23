"""Internal runtime artifact contracts for integrity diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class RuntimeArtifact:
    artifact_id: str
    kind: str
    trust_state: str
    origin: str
    source_id: str
    source_type: str
    content_hash: str
    text: str = ""
    derived_from: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    operation_metadata: Dict[str, Any] = field(default_factory=dict)
    trusted_control_excluded: bool = False

    @staticmethod
    def _trace_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        allowlist = {
            "attachment_chunk_kind",
            "attachment_modality",
            "boundary_step",
            "ocr_derived",
            "request_level_artifact",
            "request_id",
            "runtime_integrity_operation",
            "task_file",
            "visual_status",
        }
        out: Dict[str, Any] = {}
        for key in allowlist:
            if key in metadata:
                out[str(key)] = metadata.get(key)
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": str(self.artifact_id),
            "kind": str(self.kind),
            "trust_state": str(self.trust_state),
            "origin": str(self.origin),
            "source_id": str(self.source_id),
            "source_type": str(self.source_type),
            "content_hash": str(self.content_hash),
            "text_len": int(len(str(self.text or ""))),
            "derived_from": list(self.derived_from),
            "metadata": self._trace_metadata(dict(self.metadata)),
            "operation_metadata": self._trace_metadata(dict(self.operation_metadata)),
            "trusted_control_excluded": bool(self.trusted_control_excluded),
        }


@dataclass(frozen=True)
class ArtifactAssessment:
    artifact_id: str
    kind: str
    trust_state: str
    projection_status: str
    effect_signal_status: str
    integrity_signals: List[str]
    shadow_verdict: str
    would_block: bool
    requires_approval: bool
    hard_invariant_hits: List[str]
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": str(self.artifact_id),
            "kind": str(self.kind),
            "trust_state": str(self.trust_state),
            "projection_status": str(self.projection_status),
            "effect_signal_status": str(self.effect_signal_status),
            "integrity_signals": [str(x) for x in list(self.integrity_signals)],
            "shadow_verdict": str(self.shadow_verdict),
            "would_block": bool(self.would_block),
            "requires_approval": bool(self.requires_approval),
            "hard_invariant_hits": [str(x) for x in list(self.hard_invariant_hits)],
            "summary": dict(self.summary),
        }


@dataclass(frozen=True)
class OperationIntent:
    operation_type: str
    target: str
    source_artifact_ids: List[str] = field(default_factory=list)
    source_trust_states: List[str] = field(default_factory=list)
    capability_class: Optional[str] = None
    risk_level: Optional[str] = None
    approval_present: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation_type": str(self.operation_type),
            "target": str(self.target),
            "source_artifact_ids": [str(x) for x in list(self.source_artifact_ids)],
            "source_trust_states": [str(x) for x in list(self.source_trust_states)],
            "capability_class": (
                str(self.capability_class) if self.capability_class is not None else None
            ),
            "risk_level": str(self.risk_level) if self.risk_level is not None else None,
            "approval_present": bool(self.approval_present),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OperationGateDecision:
    status: str
    reason_code: str
    shadow_only: bool
    would_enforce: bool
    requires_approval: bool
    hard_invariant_hits: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": str(self.status),
            "reason_code": str(self.reason_code),
            "shadow_only": bool(self.shadow_only),
            "would_enforce": bool(self.would_enforce),
            "requires_approval": bool(self.requires_approval),
            "hard_invariant_hits": [str(x) for x in list(self.hard_invariant_hits)],
            "details": dict(self.details),
        }
