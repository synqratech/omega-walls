"""In-memory SkillBox ledger contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SkillLedgerRecord:
    skill_id: str
    skill_name: str
    source_kind: str
    requested_source_ref: str | None = None
    resolved_source_ref: str | None = None
    canonical_source_ref: str | None = None
    artifact_id: str | None = None
    content_sha256: str | None = None
    manifest_sha256: str | None = None
    capability_hash: str | None = None
    capabilities: List[str] = field(default_factory=list)
    verification_status: str = "unknown"
    approval_status: str = "unknown"
    installed_at_step: int | None = None
    last_verified_at_step: int | None = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "skill_id": str(self.skill_id),
            "skill_name": str(self.skill_name),
            "source_kind": str(self.source_kind),
            "requested_source_ref": self.requested_source_ref,
            "resolved_source_ref": self.resolved_source_ref,
            "canonical_source_ref": self.canonical_source_ref,
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "capability_hash": self.capability_hash,
            "capabilities": [str(x) for x in list(self.capabilities)],
            "verification_status": str(self.verification_status),
            "approval_status": str(self.approval_status),
            "installed_at_step": self.installed_at_step,
            "last_verified_at_step": self.last_verified_at_step,
        }


class InMemorySkillLedger:
    """Process-local ledger used for shadow verification."""

    def __init__(self) -> None:
        self._by_skill_name: Dict[str, SkillLedgerRecord] = {}
        self._by_canonical_ref: Dict[str, SkillLedgerRecord] = {}
        self._by_artifact_id: Dict[str, SkillLedgerRecord] = {}
        self._by_content_sha256: Dict[str, SkillLedgerRecord] = {}

    def lookup(
        self,
        *,
        canonical_source_ref: str | None = None,
        artifact_id: str | None = None,
        content_sha256: str | None = None,
    ) -> Optional[SkillLedgerRecord]:
        if artifact_id and artifact_id in self._by_artifact_id:
            return self._by_artifact_id[artifact_id]
        if canonical_source_ref and canonical_source_ref in self._by_canonical_ref:
            return self._by_canonical_ref[canonical_source_ref]
        if content_sha256 and content_sha256 in self._by_content_sha256:
            return self._by_content_sha256[content_sha256]
        return None

    def lookup_by_skill_name(self, *, skill_name: str | None = None) -> Optional[SkillLedgerRecord]:
        if skill_name and skill_name in self._by_skill_name:
            return self._by_skill_name[skill_name]
        return None

    def record(self, record: SkillLedgerRecord) -> None:
        skill_name = str(record.skill_name or "").strip().lower()
        canonical_ref = str(record.canonical_source_ref or "").strip()
        artifact_id = str(record.artifact_id or "").strip()
        content_sha256 = str(record.content_sha256 or "").strip()
        if skill_name:
            self._by_skill_name[skill_name] = record
        if canonical_ref:
            self._by_canonical_ref[canonical_ref] = record
        if artifact_id:
            self._by_artifact_id[artifact_id] = record
        if content_sha256:
            self._by_content_sha256[content_sha256] = record
