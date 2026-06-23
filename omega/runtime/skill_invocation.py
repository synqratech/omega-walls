"""Deterministic named-skill invocation detection for shadow diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import List, Sequence

from omega.interfaces.contracts_v1 import ContentItem


_SKILL_NAME_RE = r"([A-Za-z0-9][A-Za-z0-9._/-]{1,63})"
_SKILL_MARKER_RE = r"(?:skill|plugin|extension|connector|mcp)"
_CLAIM_RE = re.compile(
    rf"\b(?:i|we)\s+(?:have|had|installed)\s+(?:the\s+)?{_SKILL_NAME_RE}\s+{_SKILL_MARKER_RE}(?:\s+installed)?\b"
    rf"|"
    rf"\b{_SKILL_NAME_RE}\s+{_SKILL_MARKER_RE}\s+(?:is|was)\s+installed\b",
    flags=re.IGNORECASE,
)
_DIRECT_USE_RE = re.compile(
    rf"\b(?:use|apply|invoke|run|call|select)\s+(?:the\s+)?{_SKILL_NAME_RE}\s+{_SKILL_MARKER_RE}\b",
    flags=re.IGNORECASE,
)
_INSTALL_REQUEST_RE = re.compile(
    rf"\b(?:install|add|load|enable|import|fetch|download|clone)\s+(?:the\s+)?{_SKILL_NAME_RE}\s+{_SKILL_MARKER_RE}\b",
    flags=re.IGNORECASE,
)
_COREFERENCE_USE_RE = re.compile(
    r"\b(?:use it|apply it|help me with it|invoke it|run it|call it|use that skill|apply that skill)\b",
    flags=re.IGNORECASE,
)
_NEGATIVE_USE_RE = re.compile(
    r"\b(?:do not use|don't use|dont use|not to use|without using|avoid using|should not use)\b",
    flags=re.IGNORECASE,
)
_URL_INSTALL_RE = re.compile(
    r"\b(?:install|add|load|enable|import|fetch|download|clone)\s+"
    r"(https?://[^\s`'\"<>),]+)",
    flags=re.IGNORECASE,
)
_SKILL_NAME_STOPWORDS = {"the", "this", "that", "new", "installed", "skill"}


@dataclass(frozen=True)
class NamedSkillInvocationSignal:
    detected: bool
    skill_name: str | None
    invocation_type: str
    source_roles: List[str]
    source_trusts: List[str]
    confidence: float
    reason_code: str

    def to_dict(self) -> dict:
        return {
            "detected": bool(self.detected),
            "skill_name": str(self.skill_name) if self.skill_name is not None else None,
            "invocation_type": str(self.invocation_type),
            "source_roles": [str(x) for x in list(self.source_roles)],
            "source_trusts": [str(x) for x in list(self.source_trusts)],
            "confidence": float(self.confidence),
            "reason_code": str(self.reason_code),
        }


def _empty_signal() -> NamedSkillInvocationSignal:
    return NamedSkillInvocationSignal(
        detected=False,
        skill_name=None,
        invocation_type="unknown",
        source_roles=[],
        source_trusts=[],
        confidence=0.0,
        reason_code="named_skill_invocation_not_detected",
    )


def _role_for_item(item: ContentItem) -> str:
    meta = dict(getattr(item, "meta", {}) or {})
    role = str(meta.get("segment_role", "") or getattr(item, "origin", "") or "").strip().lower()
    if role:
        return role
    trust = str(getattr(item, "trust", "") or "").strip().lower()
    if trust == "trusted_user":
        return "trusted_user"
    return "untrusted"


def _eligible_items(items: Sequence[ContentItem]) -> List[ContentItem]:
    eligible: List[ContentItem] = []
    for item in list(items or []):
        trust = str(getattr(item, "trust", "untrusted") or "untrusted").strip().lower()
        if trust in {"trusted", "trusted_control"}:
            continue
        text = str(getattr(item, "text", "") or "").strip()
        if text:
            eligible.append(item)
    return eligible


def _collect_sources(items: Sequence[ContentItem]) -> tuple[List[str], List[str]]:
    roles = sorted({_role_for_item(item) for item in list(items or []) if _role_for_item(item)})
    trusts = sorted(
        {
            str(getattr(item, "trust", "untrusted") or "untrusted").strip().lower()
            for item in list(items or [])
            if str(getattr(item, "trust", "untrusted") or "untrusted").strip()
        }
    )
    return roles, trusts


def _match_skill_name(match: re.Match[str] | None) -> str | None:
    if match is None:
        return None
    for idx in range(1, (match.lastindex or 0) + 1):
        value = match.group(idx)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_skill_name(name: str | None) -> str | None:
    raw = str(name or "").strip().strip("`'\"").lower()
    if not raw or raw in _SKILL_NAME_STOPWORDS:
        return None
    return raw


def _slug_from_install_url(text: str) -> str | None:
    match = _URL_INSTALL_RE.search(str(text or ""))
    if match is None:
        return None
    raw_url = str(match.group(1)).rstrip(".,;:?!]}>)")
    path = re.sub(r"^https?://", "", raw_url, flags=re.IGNORECASE)
    parts = [part for part in path.split("/") if part]
    if not parts:
        return None
    slug = parts[-1]
    slug = re.sub(r"\.git$", "", slug, flags=re.IGNORECASE).strip().lower()
    return _normalize_skill_name(slug)


def detect_named_skill_invocation(
    items: Sequence[ContentItem],
    *,
    user_query: str = "",
) -> NamedSkillInvocationSignal:
    eligible = _eligible_items(items)
    texts = [str(getattr(item, "text", "") or "").strip() for item in eligible]
    combined = "\n\n".join(part for part in texts if part)
    if not combined and not list(items or []) and str(user_query or "").strip():
        combined = str(user_query or "").strip()
    if not combined:
        return _empty_signal()
    if _NEGATIVE_USE_RE.search(combined):
        return _empty_signal()

    roles, trusts = _collect_sources(eligible)
    direct_use = _DIRECT_USE_RE.search(combined)
    install_request = _INSTALL_REQUEST_RE.search(combined)
    claim = _CLAIM_RE.search(combined)
    install_url_slug = _slug_from_install_url(combined)
    direct_use_name = _normalize_skill_name(_match_skill_name(direct_use))
    claim_name = _normalize_skill_name(_match_skill_name(claim))
    install_request_name = _normalize_skill_name(_match_skill_name(install_request))
    if direct_use_name:
        return NamedSkillInvocationSignal(
            detected=True,
            skill_name=direct_use_name,
            invocation_type="installed_skill_use",
            source_roles=roles,
            source_trusts=trusts,
            confidence=0.92,
            reason_code="named_installed_skill_invocation",
        )
    if claim_name and _COREFERENCE_USE_RE.search(combined):
        return NamedSkillInvocationSignal(
            detected=True,
            skill_name=claim_name,
            invocation_type="installed_skill_use",
            source_roles=roles,
            source_trusts=trusts,
            confidence=0.89,
            reason_code="named_installed_skill_invocation",
        )
    if install_request_name:
        return NamedSkillInvocationSignal(
            detected=True,
            skill_name=install_request_name,
            invocation_type="skill_install_request",
            source_roles=roles,
            source_trusts=trusts,
            confidence=0.86,
            reason_code="named_skill_install_request",
        )
    if install_url_slug:
        return NamedSkillInvocationSignal(
            detected=True,
            skill_name=install_url_slug,
            invocation_type="skill_install_request",
            source_roles=roles,
            source_trusts=trusts,
            confidence=0.84,
            reason_code="named_skill_install_request",
        )
    return _empty_signal()
