"""Shadow policy gate for typed effect candidates."""

from __future__ import annotations

import re
from typing import Sequence

from omega.effects.schema import EffectPolicyGate, EffectWallCandidate
from omega.interfaces.contracts_v1 import ContentItem


SKILL_GATE_ID = "skill_effect_policy_gate_v1"
SKILL_EFFECTS = {"install_untrusted_skill", "modify_skill_or_tool"}
TRUSTED_USER_TRUSTS = {"trusted_user"}
TRUSTED_CONTROL_TRUSTS = {"trusted", "trusted_control"}
UNTRUSTED_TRUSTS = {"untrusted", "mixed", "semi", "semi_trusted", "tainted_internal"}
SKILL_MARKERS = ("skill", "plugin", "extension", "connector", "mcp", "tool")
INSTALL_VERBS = ("install", "add", "import", "enable", "load", "fetch", "download", "clone")
USE_VERBS = ("use", "invoke", "run", "activate", "call", "select")
MUTATION_VERBS = ("modify", "edit", "update", "rewrite", "patch", "override", "replace", "disable", "bypass")
INSTALL_OUTCOME_MARKERS = (
    "installed skill",
    "skill installed",
    "successfully installed",
    "install complete",
    "loaded capabilities",
    "capabilities enabled",
    "skill ready",
    "registering skill",
)
URL_RE = re.compile(r"https?://[^\s`'\"<>),]+", flags=re.IGNORECASE)


def _item_trust(item: ContentItem) -> str:
    return str(getattr(item, "trust", "untrusted") or "untrusted").strip().lower()


def _item_source_type(item: ContentItem) -> str:
    return str(getattr(item, "source_type", "unknown") or "unknown").strip().lower()


def _item_origin(item: ContentItem) -> str:
    return str(getattr(item, "origin", "") or "").strip().lower()


def _contains_any(text: str, needles: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(str(needle).lower() in lowered for needle in needles)


def _source_sets(items: Sequence[ContentItem]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    trusts = tuple(sorted({_item_trust(item) for item in list(items or []) if _item_trust(item)}))
    source_types = tuple(sorted({_item_source_type(item) for item in list(items or []) if _item_source_type(item)}))
    origins = tuple(sorted({_item_origin(item) for item in list(items or []) if _item_origin(item)}))
    return trusts, source_types, origins


def _has_trusted_user_authorization(items: Sequence[ContentItem]) -> bool:
    for item in list(items or []):
        if _item_trust(item) in TRUSTED_USER_TRUSTS:
            return True
        if _item_origin(item) in {"user", "trusted_user"} and _item_trust(item) in TRUSTED_CONTROL_TRUSTS:
            return True
    return False


def _has_untrusted_source(items: Sequence[ContentItem]) -> bool:
    if not items:
        return True
    for item in list(items or []):
        trust = _item_trust(item)
        source_type = _item_source_type(item)
        if trust in UNTRUSTED_TRUSTS:
            return True
        if "untrusted" in source_type:
            return True
    return False


def _skill_urls(text: str) -> list[str]:
    urls = []
    for raw in URL_RE.findall(str(text or "")):
        url = raw.rstrip(".,;:?!]}>)")
        lowered = url.lower()
        if any(marker in lowered for marker in ("skill", "plugin", "github.com", "gitlab.com", "bitbucket.org")):
            urls.append(lowered)
    return urls


def _has_url_mismatch(text: str) -> bool:
    urls = _skill_urls(text)
    if len(urls) < 2:
        return False
    first = urls[0]
    return any(url != first for url in urls[1:])


def _skill_urls_from_items(
    items: Sequence[ContentItem],
    *,
    trusts: Sequence[str] | None = None,
) -> list[str]:
    allowed_trusts = {str(x).strip().lower() for x in list(trusts or []) if str(x).strip()}
    urls: list[str] = []
    for item in list(items or []):
        trust = _item_trust(item)
        if allowed_trusts and trust not in allowed_trusts:
            continue
        urls.extend(_skill_urls(str(getattr(item, "text", "") or "")))
    return urls


def _has_cross_segment_url_mismatch(items: Sequence[ContentItem]) -> bool:
    trusted_urls = _skill_urls_from_items(items, trusts=TRUSTED_USER_TRUSTS | TRUSTED_CONTROL_TRUSTS)
    untrusted_urls = _skill_urls_from_items(items, trusts=UNTRUSTED_TRUSTS)
    if not trusted_urls or not untrusted_urls:
        return False
    return any(t_url != u_url for t_url in trusted_urls for u_url in untrusted_urls)


def _is_collapsed_skill_transcript(text: str) -> bool:
    lowered = str(text or "").lower()
    return _contains_any(lowered, INSTALL_OUTCOME_MARKERS) and _contains_any(
        lowered,
        ("can you install", "please install", "could you install", "install https://", "installed skill"),
    )


def _gate(
    *,
    candidate: EffectWallCandidate,
    status: str,
    reason_code: str,
    would_enforce: bool,
    source_trusts: Sequence[str],
    source_types: Sequence[str],
    rationale: str,
) -> EffectPolicyGate:
    return EffectPolicyGate(
        gate_id=SKILL_GATE_ID,
        effect=candidate.effect,
        effect_domain=candidate.effect_domain,
        status=status,
        reason_code=reason_code,
        would_enforce=bool(would_enforce),
        shadow_only=True,
        confidence=float(candidate.confidence),
        source_trusts=tuple(source_trusts),
        source_types=tuple(source_types),
        rationale=rationale,
    )


def evaluate_effect_policy_gate(
    *,
    candidate: EffectWallCandidate | None,
    text: str,
    items: Sequence[ContentItem],
) -> EffectPolicyGate | None:
    if candidate is None:
        return None
    trusts, source_types, _origins = _source_sets(items)
    if str(candidate.effect_domain) != "skill_integrity" or str(candidate.effect) not in SKILL_EFFECTS:
        return _gate(
            candidate=candidate,
            status="not_applicable",
            reason_code="effect_gate_non_skill_domain",
            would_enforce=False,
            source_trusts=trusts,
            source_types=source_types,
            rationale="Policy gate v1 only evaluates skill/tool integrity effects.",
        )

    text_l = str(text or "").lower()
    has_skill_marker = _contains_any(text_l, SKILL_MARKERS)
    has_install_verb = _contains_any(text_l, INSTALL_VERBS)
    has_use_verb = _contains_any(text_l, USE_VERBS)
    has_mutation_verb = _contains_any(text_l, MUTATION_VERBS)
    trusted_user_authorized = _has_trusted_user_authorization(items)
    untrusted_source = _has_untrusted_source(items)
    collapsed_transcript = _is_collapsed_skill_transcript(text_l)
    url_mismatch = _has_url_mismatch(text_l)
    cross_segment_url_mismatch = _has_cross_segment_url_mismatch(items)

    if trusted_user_authorized and not untrusted_source and not url_mismatch:
        return _gate(
            candidate=candidate,
            status="authorized",
            reason_code="effect_gate_trusted_user_authorized_skill_use",
            would_enforce=False,
            source_trusts=trusts,
            source_types=source_types,
            rationale="Skill effect came from trusted-user provenance without a conflicting installed source.",
        )

    if collapsed_transcript and (cross_segment_url_mismatch or url_mismatch):
        return _gate(
            candidate=candidate,
            status="passed",
            reason_code="effect_gate_skill_install_source_mismatch",
            would_enforce=True,
            source_trusts=trusts,
            source_types=source_types,
            rationale="Collapsed transcript shows a requested skill URL and a different installed skill URL.",
        )

    if collapsed_transcript:
        return _gate(
            candidate=candidate,
            status="review",
            reason_code="effect_gate_collapsed_skill_transcript_requires_provenance",
            would_enforce=False,
            source_trusts=trusts,
            source_types=source_types,
            rationale="Skill request and install outcome are collapsed into one text segment; role provenance is required before enforcement.",
        )

    if untrusted_source and has_skill_marker and candidate.effect == "install_untrusted_skill" and has_install_verb:
        return _gate(
            candidate=candidate,
            status="passed",
            reason_code="effect_gate_untrusted_skill_install_directive",
            would_enforce=True,
            source_trusts=trusts,
            source_types=source_types,
            rationale="Untrusted content is directing the agent to install or load a skill/plugin/tool.",
        )

    if untrusted_source and has_skill_marker and candidate.effect == "modify_skill_or_tool" and has_mutation_verb:
        return _gate(
            candidate=candidate,
            status="passed",
            reason_code="effect_gate_untrusted_skill_or_tool_mutation",
            would_enforce=True,
            source_trusts=trusts,
            source_types=source_types,
            rationale="Untrusted content is directing mutation of the agent skill/tool boundary.",
        )

    if untrusted_source and has_skill_marker and has_use_verb:
        return _gate(
            candidate=candidate,
            status="review",
            reason_code="effect_gate_untrusted_skill_choice_without_install_evidence",
            would_enforce=False,
            source_trusts=trusts,
            source_types=source_types,
            rationale="Untrusted content mentions skill/tool choice, but gate v1 requires install/mutation evidence before simulated enforcement.",
        )

    return _gate(
        candidate=candidate,
        status="suppressed",
        reason_code="effect_gate_skill_signal_not_policy_actionable",
        would_enforce=False,
        source_trusts=trusts,
        source_types=source_types,
        rationale="Skill effect candidate lacks policy-actionable provenance or directive evidence.",
    )
