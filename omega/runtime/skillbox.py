"""SkillBox runtime verification for skill artifacts and invocations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import zipfile

from omega.interfaces.contracts_v1 import ContentItem
from omega.runtime.skill_invocation import NamedSkillInvocationSignal, detect_named_skill_invocation
from omega.runtime.skill_ledger import InMemorySkillLedger, SkillLedgerRecord

_SOURCE_KINDS = {"url", "archive", "local_folder", "installed_skill", "tool_output", "unknown"}
_IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "tmp",
    "temp",
    ".tmp",
    ".omega_runtime",
}
_IGNORE_SUFFIXES = {".pyc", ".pyo", ".tmp", ".temp", ".lock", ".log"}
_MANIFEST_NAMES = {
    "skill.json",
    "skill.yaml",
    "skill.yml",
    "manifest.json",
    "manifest.yaml",
    "manifest.yml",
    "SKILL.md",
    "plugin.json",
}
_URL_RE = re.compile(r"https?://[^\s`'\"<>),]+", flags=re.IGNORECASE)
_REQUEST_URL_RE = re.compile(
    r"\b(?:install|add|load|enable|import|fetch|download|clone)\b[^\n]{0,120}?(https?://[^\s`'\"<>),]+)",
    flags=re.IGNORECASE,
)
_INSTALLED_FROM_RE = re.compile(
    r"(?:installed|loaded|enabled)\s+(?:skill|plugin|connector)?\s*`?([A-Za-z0-9][A-Za-z0-9._/-]{1,63})`?"
    r"(?:\s+from\s+(" + _URL_RE.pattern + r"))?",
    flags=re.IGNORECASE,
)
_RESOLVED_SOURCE_PATTERNS = (
    re.compile(
        r"\b(?:successfully\s+)?installed\s+(?:the\s+)?(?:skill|plugin|connector)"
        r"(?:\s*:\s*[`'\"]?[A-Za-z0-9][A-Za-z0-9._/-]{1,63}[`'\"]?)?"
        r"\s+from\s+(" + _URL_RE.pattern + r")",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:successfully\s+)?installed\s+(?:the\s+)?(?:skill|plugin|connector)"
        r"\s+[`'\"]?[A-Za-z0-9][A-Za-z0-9._/-]{1,63}[`'\"]?"
        r"\s+from\s+(" + _URL_RE.pattern + r")",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\binstalling\s+(?:the\s+)?(?:skill|plugin|connector)\s+from(?:\s*:)?\s+(" + _URL_RE.pattern + r")",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bsource\s*:\s*(" + _URL_RE.pattern + r")",
        flags=re.IGNORECASE,
    ),
)
_DEFAULT_SKILLBOX_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "mode": "shadow",
    "ledger_backend": "memory",
    "require_ledger_for_skill_run": True,
    "require_hash_match": True,
    "require_manifest": False,
    "enforcement": {
        "source_mismatch": False,
    },
    "dangerous_capabilities": [
        "shell_exec",
        "filesystem_write",
        "network_egress",
        "credential_access",
        "memory_write",
    ],
}


def skillbox_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    raw = config.get("skillbox", {}) if isinstance(config, Mapping) else {}
    merged = dict(_DEFAULT_SKILLBOX_CONFIG)
    merged["enforcement"] = dict(_DEFAULT_SKILLBOX_CONFIG.get("enforcement", {}))
    if isinstance(raw, Mapping):
        for key, value in dict(raw).items():
            if key == "enforcement" and isinstance(value, Mapping):
                enforcement = dict(merged.get("enforcement", {}) or {})
                enforcement.update(dict(value))
                merged["enforcement"] = enforcement
            else:
                merged[key] = value
    return merged


def skillbox_enabled(config: Mapping[str, Any]) -> bool:
    return bool(skillbox_config(config).get("enabled", False))


@dataclass(frozen=True)
class SkillArtifact:
    skill_name: str
    source_kind: str
    canonical_source_ref: str | None = None
    requested_source_ref: str | None = None
    resolved_source_ref: str | None = None
    artifact_id: str | None = None
    content_sha256: str | None = None
    manifest_sha256: str | None = None
    capability_hash: str | None = None
    capabilities: List[str] = field(default_factory=list)
    approval_present: bool = False
    derived_from: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "skill_name": str(self.skill_name),
            "source_kind": str(self.source_kind),
            "canonical_source_ref": self.canonical_source_ref,
            "requested_source_ref": self.requested_source_ref,
            "resolved_source_ref": self.resolved_source_ref,
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "capability_hash": self.capability_hash,
            "capabilities": [str(x) for x in list(self.capabilities)],
            "approval_present": bool(self.approval_present),
            "derived_from": [str(x) for x in list(self.derived_from)],
        }


@dataclass(frozen=True)
class SkillVerificationResult:
    verification_status: str
    skill_name: str
    source_kind: str
    ledger_hit: bool
    reason_code: str
    canonical_source_ref: str | None = None
    requested_source_ref: str | None = None
    resolved_source_ref: str | None = None
    artifact_id: str | None = None
    content_sha256: str | None = None
    manifest_sha256: str | None = None
    capability_hash: str | None = None
    capabilities: List[str] = field(default_factory=list)
    approval_present: bool = False
    simulated_block: bool = False
    requires_approval: bool = False

    def to_dict(self) -> Dict[str, object]:
        return {
            "verification_status": str(self.verification_status),
            "skill_name": str(self.skill_name),
            "source_kind": str(self.source_kind),
            "ledger_hit": bool(self.ledger_hit),
            "reason_code": str(self.reason_code),
            "canonical_source_ref": self.canonical_source_ref,
            "requested_source_ref": self.requested_source_ref,
            "resolved_source_ref": self.resolved_source_ref,
            "artifact_id": self.artifact_id,
            "content_sha256": self.content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "capability_hash": self.capability_hash,
            "capabilities": [str(x) for x in list(self.capabilities)],
            "approval_present": bool(self.approval_present),
            "simulated_block": bool(self.simulated_block),
            "requires_approval": bool(self.requires_approval),
        }


@dataclass(frozen=True)
class SkillInvocationCheck:
    status: str
    skill_name: str | None
    invocation_type: str
    verification: SkillVerificationResult | None
    gate_decision: str
    ledger_hit: bool = False
    would_block: bool = False
    requires_approval: bool = False
    reason_code: str = "skillbox_not_checked"

    def to_dict(self) -> Dict[str, object]:
        verification = self.verification.to_dict() if isinstance(self.verification, SkillVerificationResult) else None
        return {
            "status": str(self.status),
            "skill_name": self.skill_name,
            "invocation_type": str(self.invocation_type),
            "verification": verification,
            "gate_decision": str(self.gate_decision),
            "ledger_hit": bool(self.ledger_hit),
            "would_block": bool(self.would_block),
            "requires_approval": bool(self.requires_approval),
            "reason_code": str(self.reason_code),
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalize_capabilities(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        items = re.split(r"[,;\s]+", raw)
    elif isinstance(raw, Iterable):
        items = [str(x) for x in list(raw)]
    else:
        items = [str(raw)]
    return sorted({str(x).strip().lower() for x in items if str(x).strip()})


def _capability_hash(capabilities: Sequence[str]) -> str | None:
    caps = _normalize_capabilities(capabilities)
    if not caps:
        return None
    return _sha256_bytes(json.dumps(caps, sort_keys=True).encode("utf-8"))


def canonicalize_source_ref(source_ref: str | None) -> str | None:
    raw = str(source_ref or "").strip()
    if not raw:
        return None
    if re.match(r"^https?://", raw, flags=re.IGNORECASE):
        raw = raw.rstrip(".,;:?!]}>)")
        split = urlsplit(raw)
        query = urlencode(sorted(parse_qsl(split.query, keep_blank_values=True)))
        path = re.sub(r"/{2,}", "/", split.path or "/")
        return urlunsplit((split.scheme.lower(), split.netloc.lower(), path.rstrip("/") or "/", query, ""))
    pure = PurePosixPath(raw.replace("\\", "/"))
    return pure.as_posix().rstrip("/") or pure.as_posix()


def _should_ignore_relpath(relpath: PurePosixPath) -> bool:
    parts = {part.lower() for part in relpath.parts}
    if parts & {name.lower() for name in _IGNORE_DIRS}:
        return True
    if relpath.name.lower().startswith(".tmp"):
        return True
    if relpath.suffix.lower() in _IGNORE_SUFFIXES:
        return True
    return False


def hash_directory_tree(path: Path) -> Dict[str, str | None]:
    h = hashlib.sha256()
    manifest = hashlib.sha256()
    has_manifest = False
    for child in sorted(path.rglob("*"), key=lambda p: p.as_posix()):
        if not child.is_file():
            continue
        rel = PurePosixPath(child.relative_to(path).as_posix())
        if _should_ignore_relpath(rel):
            continue
        data = child.read_bytes()
        rel_bytes = rel.as_posix().encode("utf-8")
        h.update(len(rel_bytes).to_bytes(4, "big"))
        h.update(rel_bytes)
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
        if rel.name in _MANIFEST_NAMES:
            has_manifest = True
            manifest.update(rel_bytes)
            manifest.update(data)
    return {
        "content_sha256": h.hexdigest(),
        "manifest_sha256": manifest.hexdigest() if has_manifest else None,
    }


def hash_archive_bytes(path: Path) -> Dict[str, str | None]:
    archive_sha = _sha256_bytes(path.read_bytes())
    content_hasher = hashlib.sha256()
    manifest_hasher = hashlib.sha256()
    has_manifest = False
    try:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path, "r") as zf:
                for name in sorted(zf.namelist()):
                    rel = PurePosixPath(name)
                    if rel.name == "" or name.endswith("/") or _should_ignore_relpath(rel):
                        continue
                    data = zf.read(name)
                    rel_bytes = rel.as_posix().encode("utf-8")
                    content_hasher.update(len(rel_bytes).to_bytes(4, "big"))
                    content_hasher.update(rel_bytes)
                    content_hasher.update(len(data).to_bytes(8, "big"))
                    content_hasher.update(data)
                    if rel.name in _MANIFEST_NAMES:
                        has_manifest = True
                        manifest_hasher.update(rel_bytes)
                        manifest_hasher.update(data)
        else:
            with tarfile.open(path, "r:*") as tf:
                members = sorted(
                    (member for member in tf.getmembers() if member.isfile()),
                    key=lambda member: member.name,
                )
                for member in members:
                    rel = PurePosixPath(member.name)
                    if rel.name == "" or _should_ignore_relpath(rel):
                        continue
                    extracted = tf.extractfile(member)
                    if extracted is None:
                        continue
                    data = extracted.read()
                    rel_bytes = rel.as_posix().encode("utf-8")
                    content_hasher.update(len(rel_bytes).to_bytes(4, "big"))
                    content_hasher.update(rel_bytes)
                    content_hasher.update(len(data).to_bytes(8, "big"))
                    content_hasher.update(data)
                    if rel.name in _MANIFEST_NAMES:
                        has_manifest = True
                        manifest_hasher.update(rel_bytes)
                        manifest_hasher.update(data)
        content_sha = content_hasher.hexdigest()
    except (zipfile.BadZipFile, tarfile.TarError):
        content_sha = archive_sha
    return {
        "archive_sha256": archive_sha,
        "content_sha256": content_sha,
        "manifest_sha256": manifest_hasher.hexdigest() if has_manifest else None,
    }


def _collect_skill_metadata(
    items: Sequence[ContentItem],
    source_meta: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    if isinstance(source_meta, Mapping):
        merged.update(dict(source_meta))
        skillbox_meta = source_meta.get("skillbox")
        if isinstance(skillbox_meta, Mapping):
            merged.update(dict(skillbox_meta))
    for item in list(items or []):
        meta = dict(getattr(item, "meta", {}) or {})
        skill_meta = meta.get("skillbox")
        if isinstance(skill_meta, Mapping):
            merged.update({k: v for k, v in dict(skill_meta).items() if v not in (None, "", [])})
        for key in (
            "skill_name",
            "skill_source_kind",
            "requested_source_ref",
            "resolved_source_ref",
            "canonical_source_ref",
            "artifact_path",
            "skill_artifact_path",
            "skill_capabilities",
            "approval_present",
        ):
            value = meta.get(key)
            if value not in (None, "", []):
                merged[key] = value
    return merged


def _infer_skill_name_from_text(items: Sequence[ContentItem]) -> tuple[str | None, str | None]:
    for item in list(items or []):
        match = _INSTALLED_FROM_RE.search(str(getattr(item, "text", "") or ""))
        if match is None:
            continue
        name = str(match.group(1) or "").strip().lower()
        url = str(match.group(2) or "").strip() or None
        if name:
            return name, url
    return None, None


def _extract_resolved_source_ref_from_text(text: str) -> str | None:
    raw_text = str(text or "")
    for pattern in _RESOLVED_SOURCE_PATTERNS:
        match = pattern.search(raw_text)
        if match is not None:
            return str(match.group(1) or "").strip() or None
    installed_match = _INSTALLED_FROM_RE.search(raw_text)
    if installed_match is not None:
        return str(installed_match.group(2) or "").strip() or None
    return None


def _extract_requested_resolved_refs(
    items: Sequence[ContentItem],
    source_meta: Mapping[str, Any] | None,
) -> tuple[str | None, str | None]:
    requested = None
    resolved = None
    if isinstance(source_meta, Mapping):
        requested = source_meta.get("requested_source_ref") or source_meta.get("requested_skill_source_ref")
        resolved = source_meta.get("resolved_source_ref") or source_meta.get("resolved_skill_source_ref")
    for item in list(items or []):
        text = str(getattr(item, "text", "") or "")
        trust = str(getattr(item, "trust", "") or "").strip().lower()
        origin = str(getattr(item, "origin", "") or "").strip().lower()
        source_type = str(getattr(item, "source_type", "") or "").strip().lower()
        if requested is None and trust == "trusted_user":
            match = _REQUEST_URL_RE.search(text)
            if match is not None:
                requested = str(match.group(1) or "").strip() or requested
        if resolved is None and (source_type == "tool_output" or origin == "tool_output" or trust != "trusted_user"):
            resolved = _extract_resolved_source_ref_from_text(text) or resolved
    return canonicalize_source_ref(requested), canonicalize_source_ref(resolved)


def _first_url(text: str) -> str | None:
    match = _URL_RE.search(str(text or ""))
    if match is None:
        return None
    return str(match.group(0)).rstrip(".,;:?!]}>)")


def _extract_artifact_hashes(path_str: str | None, source_kind: str) -> Dict[str, str | None]:
    raw = str(path_str or "").strip()
    if not raw:
        return {"content_sha256": None, "manifest_sha256": None}
    path = Path(raw)
    if not path.exists():
        return {"content_sha256": None, "manifest_sha256": None}
    if path.is_dir():
        return hash_directory_tree(path)
    if path.is_file() and source_kind == "archive":
        hashed = hash_archive_bytes(path)
        return {
            "content_sha256": str(hashed.get("content_sha256") or ""),
            "manifest_sha256": hashed.get("manifest_sha256"),
        }
    data = path.read_bytes()
    manifest_sha = _sha256_bytes(data) if path.name in _MANIFEST_NAMES else None
    return {"content_sha256": _sha256_bytes(data), "manifest_sha256": manifest_sha}


def _build_compat_provenance_assessment(
    *,
    skillbox_check: SkillInvocationCheck | None,
    named_signal: NamedSkillInvocationSignal,
) -> Dict[str, Any] | None:
    if skillbox_check is None or skillbox_check.verification is None:
        return None
    return {
        "status": str(skillbox_check.verification.verification_status),
        "skill_name": str(skillbox_check.skill_name) if skillbox_check.skill_name else None,
        "invocation_type": str(skillbox_check.invocation_type),
        "reason_code": str(skillbox_check.reason_code),
        "simulated_block": bool(skillbox_check.would_block),
        "requires_approval": bool(skillbox_check.requires_approval),
        "source_trusts": list(named_signal.source_trusts),
    }


def evaluate_skillbox_shadow(
    *,
    config: Mapping[str, Any],
    items: Sequence[ContentItem],
    user_query: str = "",
    source_meta: Mapping[str, Any] | None = None,
    skillbox: SkillBox | None = None,
) -> Dict[str, Any]:
    named_skill_signal = detect_named_skill_invocation(items, user_query=user_query)
    active_skillbox = skillbox if skillbox is not None else (SkillBox.from_config(config) if skillbox_enabled(config) else None)
    if active_skillbox is None:
        return {
            "named_skill_invocation": None,
            "skill_provenance_assessment": None,
            "skillbox_status": "disabled",
            "skillbox_verification": None,
            "skillbox_ledger_hit": False,
            "skillbox_content_sha256": None,
            "skillbox_capabilities": [],
            "skillbox_gate_decision": "disabled",
        }
    if not bool(named_skill_signal.detected):
        return {
            "named_skill_invocation": named_skill_signal.to_dict(),
            "skill_provenance_assessment": None,
            "skillbox_status": "not_applicable",
            "skillbox_verification": None,
            "skillbox_ledger_hit": False,
            "skillbox_content_sha256": None,
            "skillbox_capabilities": [],
            "skillbox_gate_decision": "not_applicable",
        }
    skillbox_check = active_skillbox.check_invocation(
        items=items,
        source_meta=source_meta,
        named_signal=named_skill_signal,
    )
    return {
        "named_skill_invocation": named_skill_signal.to_dict(),
        "skill_provenance_assessment": _build_compat_provenance_assessment(
            skillbox_check=skillbox_check,
            named_signal=named_skill_signal,
        ),
        "skillbox_status": (
            str(skillbox_check.status)
            if skillbox_check is not None
            else "not_applicable"
        ),
        "skillbox_verification": (
            skillbox_check.verification.to_dict()
            if skillbox_check is not None and skillbox_check.verification is not None
            else None
        ),
        "skillbox_ledger_hit": bool(skillbox_check.ledger_hit) if skillbox_check is not None else False,
        "skillbox_content_sha256": (
            skillbox_check.verification.content_sha256
            if skillbox_check is not None and skillbox_check.verification is not None
            else None
        ),
        "skillbox_capabilities": (
            list(skillbox_check.verification.capabilities)
            if skillbox_check is not None and skillbox_check.verification is not None
            else []
        ),
        "skillbox_gate_decision": (
            str(skillbox_check.gate_decision)
            if skillbox_check is not None
            else "not_applicable"
        ),
    }


class SkillBox:
    """Default-off shadow verifier for skill artifacts."""

    def __init__(
        self,
        *,
        config: Mapping[str, Any],
        ledger: InMemorySkillLedger | None = None,
    ) -> None:
        self.config = skillbox_config(config)
        self.ledger = ledger or InMemorySkillLedger()

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        ledger: InMemorySkillLedger | None = None,
    ) -> "SkillBox":
        return cls(config=config, ledger=ledger)

    def _dangerous_capabilities(self) -> set[str]:
        return {str(x).strip().lower() for x in list(self.config.get("dangerous_capabilities", [])) if str(x).strip()}

    def build_artifact(
        self,
        *,
        items: Sequence[ContentItem],
        source_meta: Mapping[str, Any] | None,
        named_signal: NamedSkillInvocationSignal,
    ) -> SkillArtifact | None:
        if not bool(named_signal.detected):
            return None
        meta = _collect_skill_metadata(items, source_meta)
        inferred_name, inferred_resolved = _infer_skill_name_from_text(items)
        skill_name = str(
            meta.get("skill_name")
            or named_signal.skill_name
            or inferred_name
            or ""
        ).strip().lower()
        if not skill_name:
            return None
        source_kind = str(meta.get("skill_source_kind") or meta.get("source_kind") or "").strip().lower()
        if source_kind not in _SOURCE_KINDS:
            artifact_path = str(meta.get("artifact_path") or meta.get("skill_artifact_path") or "").strip()
            if artifact_path:
                path = Path(artifact_path)
                if path.is_dir():
                    source_kind = "installed_skill" if bool(meta.get("installed_skill", False)) else "local_folder"
                elif path.name.lower().endswith((".zip", ".whl", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
                    source_kind = "archive"
            elif any(_first_url(str(getattr(item, "text", "") or "")) for item in list(items or [])):
                source_kind = "url"
            elif any(str(getattr(item, "source_type", "")).strip().lower() == "tool_output" for item in list(items or [])):
                source_kind = "tool_output"
            else:
                source_kind = "installed_skill" if str(named_signal.invocation_type) == "installed_skill_use" else "unknown"
        parsed_requested_source_ref, parsed_resolved_source_ref = _extract_requested_resolved_refs(
            items,
            source_meta,
        )
        requested_source_ref = canonicalize_source_ref(
            meta.get("requested_source_ref")
            or meta.get("requested_skill_source_ref")
            or parsed_requested_source_ref
        )
        resolved_source_ref = canonicalize_source_ref(
            meta.get("resolved_source_ref")
            or meta.get("resolved_skill_source_ref")
            or inferred_resolved
            or parsed_resolved_source_ref
        )
        canonical_source_ref = canonicalize_source_ref(
            meta.get("canonical_source_ref") or resolved_source_ref or requested_source_ref
        )
        artifact_id = str(meta.get("artifact_id") or meta.get("skill_artifact_id") or "").strip() or None
        capabilities = _normalize_capabilities(meta.get("skill_capabilities") or meta.get("capabilities"))
        capability_hash = _capability_hash(capabilities)
        approval_present = bool(meta.get("approval_present", False))
        derived_from: List[str] = []
        for item in list(items or []):
            for value in list(getattr(item, "derived_from", None) or []):
                if str(value).strip():
                    derived_from.append(str(value))
        path_str = str(meta.get("artifact_path") or meta.get("skill_artifact_path") or "").strip() or None
        hashes = _extract_artifact_hashes(path_str, source_kind)
        return SkillArtifact(
            skill_name=skill_name,
            source_kind=source_kind,
            canonical_source_ref=canonical_source_ref,
            requested_source_ref=requested_source_ref,
            resolved_source_ref=resolved_source_ref,
            artifact_id=artifact_id,
            content_sha256=hashes.get("content_sha256"),
            manifest_sha256=hashes.get("manifest_sha256"),
            capability_hash=capability_hash,
            capabilities=capabilities,
            approval_present=approval_present,
            derived_from=sorted(set(derived_from)),
        )

    def verify(self, artifact: SkillArtifact) -> SkillVerificationResult:
        record = self.ledger.lookup(
            canonical_source_ref=artifact.canonical_source_ref,
            artifact_id=artifact.artifact_id,
            content_sha256=artifact.content_sha256,
        )
        ledger_hit = record is not None
        status = "verified"
        reason_code = "skillbox_verified"
        simulated_block = False
        requires_approval = False
        dangerous = bool(set(artifact.capabilities) & self._dangerous_capabilities())
        ambiguous_name_match = (
            self.ledger.lookup_by_skill_name(skill_name=artifact.skill_name) is not None
            and not ledger_hit
        )
        if artifact.requested_source_ref and artifact.resolved_source_ref:
            if artifact.requested_source_ref != artifact.resolved_source_ref:
                status = "source_mismatch"
                reason_code = "skillbox_source_mismatch"
                simulated_block = True
        if status == "verified" and dangerous and not bool(artifact.approval_present):
            status = "dangerous_capability_unapproved"
            reason_code = "skillbox_dangerous_capability_unapproved"
            simulated_block = True
            requires_approval = True
        elif status == "verified" and not ledger_hit and bool(
            self.config.get("require_ledger_for_skill_run", True)
        ):
            status = "unknown"
            reason_code = "skillbox_name_only_match_requires_verification" if ambiguous_name_match else "skillbox_missing_ledger"
            requires_approval = True
        if status == "verified" and bool(self.config.get("require_manifest", False)) and not artifact.manifest_sha256:
            status = "missing_manifest"
            reason_code = "skillbox_missing_manifest"
            requires_approval = True
        if status == "verified" and ledger_hit and record is not None:
            if bool(self.config.get("require_hash_match", True)):
                if artifact.content_sha256 and not record.content_sha256:
                    status = "unknown"
                    reason_code = "skillbox_ledger_missing_hash"
                    requires_approval = True
                if artifact.content_sha256 and record.content_sha256 and artifact.content_sha256 != record.content_sha256:
                    status = "tampered" if artifact.source_kind in {"installed_skill", "local_folder"} else "hash_mismatch"
                    reason_code = "skillbox_tampered" if status == "tampered" else "skillbox_hash_mismatch"
                    simulated_block = True
                elif artifact.capability_hash and record.capability_hash and artifact.capability_hash != record.capability_hash:
                    status = "tampered"
                    reason_code = "skillbox_tampered"
                    simulated_block = True
            if (
                status == "verified"
                and artifact.resolved_source_ref
                and record.resolved_source_ref
                and artifact.resolved_source_ref != canonicalize_source_ref(record.resolved_source_ref)
            ):
                status = "source_mismatch"
                reason_code = "skillbox_source_mismatch"
                simulated_block = True
        if status == "verified" and not ledger_hit:
            status = "unknown"
            reason_code = "skillbox_name_only_match_requires_verification" if ambiguous_name_match else "skillbox_unknown"
            requires_approval = True
        return SkillVerificationResult(
            verification_status=status,
            skill_name=artifact.skill_name,
            source_kind=artifact.source_kind,
            ledger_hit=ledger_hit,
            reason_code=reason_code,
            canonical_source_ref=artifact.canonical_source_ref,
            requested_source_ref=artifact.requested_source_ref,
            resolved_source_ref=artifact.resolved_source_ref,
            artifact_id=artifact.artifact_id,
            content_sha256=artifact.content_sha256,
            manifest_sha256=artifact.manifest_sha256,
            capability_hash=artifact.capability_hash,
            capabilities=list(artifact.capabilities),
            approval_present=bool(artifact.approval_present),
            simulated_block=simulated_block,
            requires_approval=requires_approval,
        )

    def gate(self, verification: SkillVerificationResult) -> str:
        status = str(verification.verification_status)
        if status == "verified" and bool(verification.approval_present or not verification.requires_approval):
            return "allow"
        if status in {"source_mismatch", "hash_mismatch", "tampered"}:
            return "would_block"
        if status == "dangerous_capability_unapproved":
            return "require_approval"
        if status in {"unknown", "missing_manifest"}:
            return "review"
        return "shadow_only"

    def check_invocation(
        self,
        *,
        items: Sequence[ContentItem],
        source_meta: Mapping[str, Any] | None,
        named_signal: NamedSkillInvocationSignal,
    ) -> SkillInvocationCheck | None:
        if not bool(named_signal.detected):
            return None
        artifact = self.build_artifact(items=items, source_meta=source_meta, named_signal=named_signal)
        if artifact is None:
            return None
        verification = self.verify(artifact)
        gate_decision = self.gate(verification)
        return SkillInvocationCheck(
            status="checked",
            skill_name=artifact.skill_name,
            invocation_type=str(named_signal.invocation_type),
            verification=verification,
            gate_decision=gate_decision,
            ledger_hit=bool(verification.ledger_hit),
            would_block=bool(verification.simulated_block),
            requires_approval=bool(verification.requires_approval),
            reason_code=str(verification.reason_code),
        )

    def seed_record(self, artifact: SkillArtifact, *, verification_status: str = "verified") -> None:
        self.ledger.record(
            SkillLedgerRecord(
                skill_id=str(artifact.skill_name),
                skill_name=str(artifact.skill_name),
                source_kind=str(artifact.source_kind),
                requested_source_ref=artifact.requested_source_ref,
                resolved_source_ref=artifact.resolved_source_ref,
                canonical_source_ref=artifact.canonical_source_ref,
                artifact_id=artifact.artifact_id,
                content_sha256=artifact.content_sha256,
                manifest_sha256=artifact.manifest_sha256,
                capability_hash=artifact.capability_hash,
                capabilities=list(artifact.capabilities),
                verification_status=verification_status,
                approval_status=("approved" if artifact.approval_present else "unknown"),
            )
        )
