from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from omega.projector.normalize import tokenize
from omega.projector.pi0_intent_v2 import (
    PI0_POLICY_EVASION_DEFENSIVE_AMBIGUITY_MARKERS,
    _canonicalize_tokens,
    _contains_any,
    _near_negation_with_markers,
    _pair_within_window,
)

_BUNDLE_EXCEPTION_PHRASES: Tuple[str, ...] = (
    "activate exception route",
    "exception route is temporarily active",
    "defer baseline safeguards",
    "standard hold window may be skipped",
    "reconciliation override is currently enabled",
)
_BUNDLE_WORKFLOW_CONTEXT_TOKENS: Tuple[str, ...] = (
    "beneficiary",
    "settlement",
    "transfer",
    "vendor",
    "payout",
    "finance",
    "reconciliation",
    "routing",
    "route",
    "lane",
    "branch",
    "workflow",
    "queue",
    "handoff",
    "verification",
    "case",
)
_BUNDLE_CONTINUATION_MARKERS: Tuple[str, ...] = (
    "carry-forward",
    "carry forward",
    "carryover",
    "handoff",
    "follow-up",
    "follow up",
    "prior note",
    "prior records",
    "from earlier",
    "previous session",
    "latest context",
    "retained from earlier",
    "persisted",
    "active context",
)
_BUNDLE_PROTECTIVE_NEGATORS: Tuple[str, ...] = (
    "prevent",
    "prevents",
    "mitigate",
    "mitigates",
    "reduce",
    "reduces",
)
_BUNDLE_PROTECTIVE_PHRASES: Tuple[str, ...] = (
    "how to prevent policy bypass",
    "explains how to prevent policy bypass",
    "prevent policy bypass",
    "policy bypass prevention",
    "security documentation",
    "security guide",
)
_BUNDLE_ROUTE_EXCEPTION_LEFT: Tuple[str, ...] = (
    "exception",
    "policy",
    "route",
    "routing",
    "hold",
    "reconciliation",
    "override",
)
_BUNDLE_ROUTE_EXCEPTION_RIGHT: Tuple[str, ...] = (
    "route",
    "active",
    "enabled",
    "skipped",
    "waived",
    "safeguards",
    "guardrails",
)


def _count_token_hits(tokens: Sequence[str], vocab: Sequence[str]) -> int:
    vocab_set = set(vocab)
    return sum(1 for tok in tokens if tok in vocab_set)


def evasion_score(self: Any, t: str, struct_count: int) -> Tuple[float, Dict[str, Any], int]:
    score, matches, pol = self._evasion_score_legacy(t, struct_count)

    tokens = _canonicalize_tokens(tokenize(t))
    neg_near = _near_negation_with_markers(
        tokens,
        list(set(self.dict_cfg.get("evasion_verbs", [])) | set(self.dict_cfg.get("evasion_objects", []))),
        int(self.pi0_cfg["win_neg"]),
        ["do not", "don't", "dont", "never", "avoid", "refuse", *_BUNDLE_PROTECTIVE_NEGATORS],
    )
    defensive_ambiguity = bool(
        _contains_any(t, PI0_POLICY_EVASION_DEFENSIVE_AMBIGUITY_MARKERS)
        or _contains_any(t, _BUNDLE_PROTECTIVE_PHRASES)
        or _contains_any(
            t,
            (
                "security guidance",
                "defensive guidance",
                "training example",
                "example only",
                "documentation",
                "tutorial",
                "treat it as hostile",
                "must be refused",
            ),
        )
    )

    phrase_hits = [phrase for phrase in _BUNDLE_EXCEPTION_PHRASES if phrase in t]
    route_pair_hit, route_pair_windows = _pair_within_window(
        tokens,
        _BUNDLE_ROUTE_EXCEPTION_LEFT,
        _BUNDLE_ROUTE_EXCEPTION_RIGHT,
        max(4, int(self.pi0_cfg["win_pair"])),
    )
    workflow_context_hits = _count_token_hits(tokens, _BUNDLE_WORKFLOW_CONTEXT_TOKENS)
    continuation_hits: List[str] = [marker for marker in _BUNDLE_CONTINUATION_MARKERS if marker in t]
    bundle_ready = bool(
        (phrase_hits or route_pair_hit)
        and workflow_context_hits >= 2
        and continuation_hits
        and (not defensive_ambiguity)
        and (not neg_near)
    )

    base_matches = matches if isinstance(matches, dict) else {}
    markers = list(base_matches.get("markers", []) or [])
    windows = list(base_matches.get("windows", []) or [])

    if bundle_ready:
        bonus = float(self.weights["evasion"].get("w_pair", 0.0))
        score += bonus
        markers.extend(hit for hit in phrase_hits if hit not in markers)
        if route_pair_hit:
            windows.extend(route_pair_windows)
        if pol == 0:
            pol = 1

    enriched_matches = {
        **base_matches,
        "markers": markers,
        "windows": windows,
        "bundle_phrase_hits": list(phrase_hits),
        "bundle_route_pair_hit": bool(route_pair_hit),
        "bundle_workflow_context_hits": int(workflow_context_hits),
        "bundle_continuation_hits": list(continuation_hits),
        "bundle_promotion_applied": bool(bundle_ready),
        "bundle_defensive_ambiguity": bool(defensive_ambiguity),
    }
    return score, enriched_matches, pol
