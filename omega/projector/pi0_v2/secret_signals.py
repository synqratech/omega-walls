from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from omega.projector.normalize import tokenize
from omega.projector.pi0_intent_v2 import (
    PI0_LOCAL_AMBIGUITY_MARKERS,
    _canonicalize_tokens,
    _contains_any,
    _pair_within_window,
)

_SECRET_OBJECT_PHRASES: Tuple[str, ...] = (
    "api token",
    "api key",
    "auth token",
    "access token",
    "credential",
    "credentials",
)
_SECRET_DELIVERY_PHRASES: Tuple[str, ...] = (
    "include verification payload snippet",
    "embed credential-adjacent diagnostics for confirmation",
    "attach service context for expedited validation",
)
_SECRET_DELIVERY_VERBS: Tuple[str, ...] = (
    "attach",
    "include",
    "embed",
    "export",
    "send",
    "share",
    "transmit",
    "present",
    "return",
)
_SECRET_VERIFICATION_CONTEXT_TOKENS: Tuple[str, ...] = (
    "verification",
    "validate",
    "validation",
    "confirmation",
    "payload",
    "snippet",
    "diagnostics",
    "service",
    "context",
    "response",
    "evidence",
    "triage",
    "audit",
)
_SECRET_BUNDLE_PROTECTIVE_PHRASES: Tuple[str, ...] = (
    "security documentation",
    "security guide",
    "documentation example",
    "training example",
    "example only",
    "do not expose api token",
    "do not return api token",
)


def _count_token_hits(tokens: Sequence[str], vocab: Sequence[str]) -> int:
    vocab_set = set(vocab)
    return sum(1 for tok in tokens if tok in vocab_set)


def secret_score(self: Any, t: str, tokens: Sequence[str], struct_count: int) -> Tuple[float, Dict[str, Any], int, bool]:
    score, matches, pol, exfil_like_intent = self._secret_score_legacy(t, tokens, struct_count)

    canon_tokens = _canonicalize_tokens(tokens or tokenize(t))
    secret_phrase_hits = [phrase for phrase in _SECRET_OBJECT_PHRASES if phrase in t]
    delivery_phrase_hits = [phrase for phrase in _SECRET_DELIVERY_PHRASES if phrase in t]
    delivery_pair_hit, delivery_windows = _pair_within_window(
        canon_tokens,
        _SECRET_DELIVERY_VERBS,
        tuple(str(x) for x in self.dict_cfg.get("secret_nouns", [])),
        max(4, int(self.pi0_cfg["win_pair"])),
    )
    verification_context_hits = _count_token_hits(canon_tokens, _SECRET_VERIFICATION_CONTEXT_TOKENS)
    defensive_ambiguity = bool(
        _contains_any(t, PI0_LOCAL_AMBIGUITY_MARKERS)
        or _contains_any(t, _SECRET_BUNDLE_PROTECTIVE_PHRASES)
        or _contains_any(
            t,
            (
                "malicious example",
                "quoted malicious",
                "not active instruction",
                "should not be treated as an active instruction",
                "should not be treated as attack pressure",
            ),
        )
    )

    strong_quote_ambiguity = bool(
        _contains_any(
            t,
            (
                "training quote",
                "quoted malicious",
                "quoted example",
                "example only",
                "not an active instruction",
                "not active instruction",
            ),
        )
    )
    if strong_quote_ambiguity:
        score = 0.0
        pol = 0
        exfil_like_intent = False

    bundle_ready = bool(
        secret_phrase_hits
        and (delivery_phrase_hits or delivery_pair_hit)
        and verification_context_hits >= 2
        and not defensive_ambiguity
        and pol == 0
    )

    base_matches = matches if isinstance(matches, dict) else {}
    windows = list(base_matches.get("windows", []) or [])
    if delivery_pair_hit:
        windows.extend(delivery_windows)

    enriched_matches = {
        **base_matches,
        "windows": windows,
        "secret_bundle_secret_hits": list(secret_phrase_hits),
        "secret_bundle_delivery_phrase_hits": list(delivery_phrase_hits),
        "secret_bundle_delivery_pair_hit": bool(delivery_pair_hit),
        "secret_bundle_verification_context_hits": int(verification_context_hits),
        "secret_bundle_promotion_applied": bool(bundle_ready),
        "secret_bundle_defensive_ambiguity": bool(defensive_ambiguity),
    }
    if bundle_ready:
        pol = 1
        exfil_like_intent = True

    return score, enriched_matches, pol, exfil_like_intent
