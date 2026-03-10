"""Token-level gapped rule engine (DNF-lite) for intent detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class GappedMatch:
    matched: bool
    rule_ids: List[str]
    matched_clause_ids: List[str]
    windows: List[Dict[str, Any]]
    negation_blocked_rule_ids: List[str]
    bonus: float


def _token_positions(tokens: Sequence[str], vocab: Iterable[str]) -> Dict[str, List[int]]:
    vocab_set = set(str(v).strip().lower() for v in vocab if str(v).strip())
    out: Dict[str, List[int]] = {}
    for idx, tok in enumerate(tokens):
        if tok in vocab_set:
            out.setdefault(tok, []).append(idx)
    return out


def _pair_matches(tokens: Sequence[str], left_any: Sequence[str], right_any: Sequence[str], max_gap: int) -> List[Dict[str, Any]]:
    left_pos = _token_positions(tokens, left_any)
    right_pos = _token_positions(tokens, right_any)
    out: List[Dict[str, Any]] = []
    for ltok, li_list in left_pos.items():
        for rtok, ri_list in right_pos.items():
            for li in li_list:
                for ri in ri_list:
                    dist = abs(li - ri)
                    if dist <= max_gap:
                        out.append(
                            {
                                "kind": "pair",
                                "left": ltok,
                                "right": rtok,
                                "left_idx": li,
                                "right_idx": ri,
                                "dist": dist,
                                "max_gap": int(max_gap),
                            }
                        )
    return out


def _token_any_matches(tokens: Sequence[str], vocab_any: Sequence[str], min_hits: int) -> List[Dict[str, Any]]:
    vocab = set(str(v).strip().lower() for v in vocab_any if str(v).strip())
    if not vocab:
        return []
    out: List[Dict[str, Any]] = []
    for idx, tok in enumerate(tokens):
        if tok in vocab:
            out.append({"kind": "token_any", "token": tok, "idx": idx, "min_hits": int(min_hits)})
    if len(out) < int(min_hits):
        return []
    return out


def _marker_in_window(window_tokens: Sequence[str], marker: str) -> bool:
    marker_tokens = [t for t in str(marker).split() if t]
    if not marker_tokens:
        return False
    if len(marker_tokens) == 1:
        return marker_tokens[0] in window_tokens
    for i in range(0, max(0, len(window_tokens) - len(marker_tokens) + 1)):
        if list(window_tokens[i : i + len(marker_tokens)]) == marker_tokens:
            return True
    return False


def _has_negation_near(tokens: Sequence[str], positions: Sequence[int], neg_markers: Sequence[str], window: int) -> bool:
    if not positions:
        return False
    for idx in positions:
        left = max(0, int(idx) - int(window))
        right = min(len(tokens), int(idx) + int(window) + 1)
        span = tokens[left:right]
        for marker in neg_markers:
            if _marker_in_window(span, str(marker)):
                return True
    return False


def _must_include_ok(tokens: Sequence[str], must_include_any: Sequence[str]) -> bool:
    if not must_include_any:
        return True
    token_set = set(tokens)
    required = set(str(v).strip().lower() for v in must_include_any if str(v).strip())
    return len(token_set & required) > 0


def evaluate_gapped_rules(
    *,
    tokens: Sequence[str],
    cfg: Dict[str, Any] | None,
    neg_markers: Sequence[str],
) -> GappedMatch:
    data = cfg or {}
    if not bool(data.get("enabled", False)):
        return GappedMatch(
            matched=False,
            rule_ids=[],
            matched_clause_ids=[],
            windows=[],
            negation_blocked_rule_ids=[],
            bonus=0.0,
        )

    rules = data.get("rules", [])
    if not isinstance(rules, list):
        rules = []

    global_neg_window = int(data.get("negation_window", 6))
    default_bonus = float(data.get("default_rule_bonus", 0.0))
    bonus_cap = float(data.get("gapped_bonus_cap", 0.0))
    neg_vocab = [str(v).strip().lower() for v in (data.get("neg_markers", neg_markers) or []) if str(v).strip()]

    matched_rule_ids: List[str] = []
    matched_clause_ids: List[str] = []
    matched_windows: List[Dict[str, Any]] = []
    blocked_rule_ids: List[str] = []
    bonus = 0.0

    for ridx, rule in enumerate(rules):
        if not isinstance(rule, dict):
            continue
        rule_id = str(rule.get("id", f"rule_{ridx}"))
        if not _must_include_ok(tokens, rule.get("must_include_any", [])):
            continue

        clauses = rule.get("clauses", [])
        if not isinstance(clauses, list) or not clauses:
            continue
        per_rule_neg_window = int(rule.get("negation_window", global_neg_window))
        per_rule_neg_markers = [
            str(v).strip().lower() for v in (rule.get("neg_markers", neg_vocab) or []) if str(v).strip()
        ]
        rule_bonus = float(rule.get("bonus", default_bonus))

        rule_matched = False
        for cidx, clause in enumerate(clauses):
            if not isinstance(clause, dict):
                continue
            clause_id = str(clause.get("id", f"{rule_id}:clause_{cidx}"))
            conditions = clause.get("all", [])
            if not isinstance(conditions, list) or not conditions:
                continue

            clause_windows: List[Dict[str, Any]] = []
            all_ok = True
            for cond in conditions:
                if not isinstance(cond, dict):
                    all_ok = False
                    break
                kind = str(cond.get("kind", "")).strip().lower()
                if kind == "pair":
                    cw = _pair_matches(
                        tokens=tokens,
                        left_any=cond.get("left_any", []),
                        right_any=cond.get("right_any", []),
                        max_gap=int(cond.get("max_gap", 4)),
                    )
                    if not cw:
                        all_ok = False
                        break
                    clause_windows.extend(cw)
                elif kind == "token_any":
                    cw = _token_any_matches(
                        tokens=tokens,
                        vocab_any=cond.get("vocab_any", []),
                        min_hits=int(cond.get("min_hits", 1)),
                    )
                    if not cw:
                        all_ok = False
                        break
                    clause_windows.extend(cw)
                else:
                    all_ok = False
                    break

            if not all_ok:
                continue

            positions: List[int] = []
            for w in clause_windows:
                if "idx" in w:
                    positions.append(int(w["idx"]))
                if "left_idx" in w:
                    positions.append(int(w["left_idx"]))
                if "right_idx" in w:
                    positions.append(int(w["right_idx"]))
            if _has_negation_near(
                tokens=tokens,
                positions=positions,
                neg_markers=per_rule_neg_markers,
                window=per_rule_neg_window,
            ):
                blocked_rule_ids.append(rule_id)
                continue

            rule_matched = True
            matched_rule_ids.append(rule_id)
            matched_clause_ids.append(clause_id)
            matched_windows.extend([{"rule_id": rule_id, "clause_id": clause_id, **w} for w in clause_windows])
            bonus += max(0.0, rule_bonus)
            break

        if not rule_matched:
            continue

    if bonus_cap > 0.0:
        bonus = min(bonus, bonus_cap)

    return GappedMatch(
        matched=len(matched_rule_ids) > 0,
        rule_ids=sorted(set(matched_rule_ids)),
        matched_clause_ids=matched_clause_ids,
        windows=matched_windows,
        negation_blocked_rule_ids=sorted(set(blocked_rule_ids)),
        bonus=float(bonus),
    )
