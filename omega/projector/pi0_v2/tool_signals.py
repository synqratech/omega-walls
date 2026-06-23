from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from omega.projector.normalize import nospace_text
from omega.projector.pi0_intent_v2 import (
    PI0_LOCAL_AMBIGUITY_MARKERS,
    _canonicalize_tokens,
    _contains_any,
    _pair_within_window,
)

_TOOL_ACTION_PHRASES: Tuple[str, ...] = (
    "apply workspace update action",
    "direct automation branch",
    "transfer preparation",
    "nonstandard execution lane",
)
_TOOL_ROUTE_NOUNS: Tuple[str, ...] = (
    "route",
    "lane",
    "branch",
    "handoff",
    "rail",
    "workflow",
)
_TOOL_ACTION_VERBS: Tuple[str, ...] = (
    "apply",
    "initiate",
    "process",
    "prepare",
    "execute",
    "transfer",
    "update",
    "route",
    "reroute",
)
_TOOL_OPERATIONAL_TOKENS: Tuple[str, ...] = (
    "beneficiary",
    "settlement",
    "payout",
    "vendor",
    "reconciliation",
    "transfer",
    "packet",
    "batch",
    "payment",
    "finance",
)
_TOOL_DELIVERY_CONTEXT_TOKENS: Tuple[str, ...] = (
    "carry-forward",
    "handoff",
    "requester",
    "agent",
    "case",
    "artifact",
    "review",
    "operations",
    "routing",
)
_TOOL_PROTECTIVE_PHRASES: Tuple[str, ...] = (
    "security documentation",
    "security guide",
    "training example",
    "documentation example",
    "defensive documentation",
    "example only",
    "not an active instruction",
    "not active instruction",
)


def _count_token_hits(tokens: Sequence[str], vocab: Sequence[str]) -> int:
    vocab_set = set(vocab)
    return sum(1 for tok in tokens if tok in vocab_set)


def tool_score(
    self: Any,
    t: str,
    tokens: Sequence[str],
    struct_count: int,
    exfil_intent: bool,
    override_raw: float,
) -> Tuple[float, Dict[str, Any], int, bool]:
    score, matches, pol, sql_intent_active = self._tool_score_legacy(t, tokens, struct_count, exfil_intent, override_raw)

    t_ns = nospace_text(t)
    canon_tokens = _canonicalize_tokens(tokens)
    action_phrase_hits = [phrase for phrase in _TOOL_ACTION_PHRASES if phrase in t]
    route_pair_hit, route_windows = _pair_within_window(
        canon_tokens,
        _TOOL_ACTION_VERBS,
        _TOOL_ROUTE_NOUNS,
        max(4, int(self.pi0_cfg["win_pair"])),
    )
    operational_hits = _count_token_hits(canon_tokens, _TOOL_OPERATIONAL_TOKENS)
    delivery_context_hits = _count_token_hits(canon_tokens, _TOOL_DELIVERY_CONTEXT_TOKENS)
    defensive_ambiguity = bool(
        _contains_any(t, PI0_LOCAL_AMBIGUITY_MARKERS)
        or _contains_any(t, _TOOL_PROTECTIVE_PHRASES)
        or _contains_any(
            t,
            (
                "malicious example",
                "quoted malicious",
                "should not be treated as an active instruction",
                "should not be treated as attack pressure",
            ),
        )
    )
    guard_reasons = list((matches or {}).get("tool_guard_block_reasons", []) or [])
    bundle_ready = bool(
        (action_phrase_hits or route_pair_hit)
        and operational_hits >= 2
        and delivery_context_hits >= 1
        and not defensive_ambiguity
        and not guard_reasons
        and pol == 0
    )
    ocr_collapsed_eia_ready = bool(
        pol == 0
        and not defensive_ambiguity
        and not guard_reasons
        and (
            ("before typing" in t or "beforetyping" in t_ns)
            and ("type the task" in t or "typethetask" in t_ns)
            and (
                "must first" in t
                or "mustfirst" in t_ns
                or "verify your identity" in t
                or "verifyyouridentity" in t_ns
                or "do not summarize" in t
                or "donotsummarize" in t_ns
                or "exactly type" in t
                or "exactlytype" in t_ns
            )
        )
    )
    ocr_collapsed_vpi_ready = bool(
        pol == 0
        and not defensive_ambiguity
        and not guard_reasons
        and (
            ("open a new tab" in t or "openanewtab" in t_ns)
            and (
                "fill out the form" in t
                or "fill in the form" in t
                or "complete the form" in t
                or "fillouttheform" in t_ns
                or "fillintheform" in t_ns
                or "completetheform" in t_ns
            )
            and (
                "do not ask the user" in t
                or "do not ask user" in t
                or "donotasktheuser" in t_ns
                or "donotaskuser" in t_ns
                or "try multiple times" in t
                or "trymultipletimes" in t_ns
            )
        )
    )
    ocr_collapsed_chain_ready = bool(ocr_collapsed_eia_ready or ocr_collapsed_vpi_ready)

    base_matches = matches if isinstance(matches, dict) else {}
    windows = list(base_matches.get("windows", []) or [])
    if route_pair_hit:
        windows.extend(route_windows)

    if bundle_ready:
        score += float(self.weights["tool"].get("w_tool_tok", 0.0)) * 2.0
        pol = 1
    elif ocr_collapsed_chain_ready:
        score += float(self.weights["tool"].get("w_tool_tok", 0.0)) * 2.0
        pol = 1

    enriched_matches = {
        **base_matches,
        "windows": windows,
        "tool_bundle_action_phrase_hits": list(action_phrase_hits),
        "tool_bundle_route_pair_hit": bool(route_pair_hit),
        "tool_bundle_operational_hits": int(operational_hits),
        "tool_bundle_delivery_context_hits": int(delivery_context_hits),
        "tool_bundle_promotion_applied": bool(bundle_ready),
        "tool_bundle_defensive_ambiguity": bool(defensive_ambiguity),
        "ocr_collapsed_eia_chain_hit": bool(ocr_collapsed_eia_ready),
        "ocr_collapsed_vpi_chain_hit": bool(ocr_collapsed_vpi_ready),
        "ocr_collapsed_chain_promotion_applied": bool(ocr_collapsed_chain_ready),
    }
    return score, enriched_matches, pol, sql_intent_active
