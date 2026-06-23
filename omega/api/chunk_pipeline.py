"""Chunk-level scoring pipeline for attachment scan API.

Pipeline:
1. Score each chunk with projector (pi0 / hybrid / pitheta).
2. Aggregate at document level:
   - worst chunk score
   - pattern synergy
   - confidence
3. Produce deterministic doc_score in [0,1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple


from omega.interfaces.contracts_v1 import ContentItem, ProjectionResult


DEFAULT_SYNERGY_PAIRS: List[Tuple[str, str]] = [
    ("override_instructions", "tool_or_action_abuse"),
    ("override_instructions", "secret_exfiltration"),
    ("policy_evasion", "override_instructions"),
]


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


@dataclass(frozen=True)
class ChunkPipelineConfig:
    wall_trigger_threshold: float = 0.12
    worst_weight: float = 0.65
    synergy_weight: float = 0.20
    confidence_weight: float = 0.15
    synergy_pair_bonus: float = 0.18
    synergy_multiwall_bonus: float = 0.10
    synergy_pattern_bonus: float = 0.05
    confidence_margin: float = 0.15
    confidence_support_threshold: float = 0.35
    confidence_support_chunks: int = 2
    top_chunks_limit: int = 3

    @classmethod
    def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "ChunkPipelineConfig":
        data = dict(cfg or {})
        return cls(
            wall_trigger_threshold=float(data.get("wall_trigger_threshold", 0.12)),
            worst_weight=float(data.get("worst_weight", 0.65)),
            synergy_weight=float(data.get("synergy_weight", 0.20)),
            confidence_weight=float(data.get("confidence_weight", 0.15)),
            synergy_pair_bonus=float(data.get("synergy_pair_bonus", 0.18)),
            synergy_multiwall_bonus=float(data.get("synergy_multiwall_bonus", 0.10)),
            synergy_pattern_bonus=float(data.get("synergy_pattern_bonus", 0.05)),
            confidence_margin=float(data.get("confidence_margin", 0.15)),
            confidence_support_threshold=float(data.get("confidence_support_threshold", 0.35)),
            confidence_support_chunks=int(data.get("confidence_support_chunks", 2)),
            top_chunks_limit=int(data.get("top_chunks_limit", 3)),
        )


@dataclass(frozen=True)
class ChunkScore:
    doc_id: str
    score_max: float
    wall_scores: Dict[str, float]
    active_walls: List[str]
    pattern_signals: List[str]
    matched_rule_ids: List[str]


@dataclass(frozen=True)
class ChunkAggregationResult:
    chunk_scores: List[ChunkScore]
    projections: List[ProjectionResult]
    wall_max: Dict[str, float]
    worst_chunk_score: float
    pattern_synergy: float
    confidence: float
    doc_score: float
    pair_hits: List[str]
    top_chunks: List[Dict[str, Any]]
    rule_ids: List[str]
    triggered_chunk_ids: List[str]
    reasons: List[str]


def _extract_pattern_signals(matches: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    if matches.get("gapped_rule_ids"):
        out.append("gapped_override")
    if matches.get("secret_leak_markers"):
        out.append("secret_leak_marker")
    if bool(matches.get("sql_db_pair_intent", False)):
        out.append("sql_db_pair_intent")
    if matches.get("evasion_markers"):
        out.append("evasion_marker")
    if int(matches.get("decoded_segments_count", 0)) > 0:
        out.append("decoded_segments")
    if int(matches.get("joined_obfuscation_sequences_count", 0)) > 0:
        out.append("joined_obfuscation")
    if matches.get("preprocess_context_kinds"):
        out.append("preprocess_contexts")
    return sorted(set(out))


def _extract_rule_ids(matches: Mapping[str, Any]) -> List[str]:
    rule_ids: List[str] = []
    for rid in (matches.get("gapped_rule_ids") or []):
        rid_s = str(rid).strip()
        if rid_s:
            rule_ids.append(f"gapped:{rid_s}")
    if bool(matches.get("sql_db_pair_intent", False)):
        rule_ids.append("tool:sql_db_pair_intent")
    if matches.get("secret_leak_markers"):
        rule_ids.append("secret:leak_marker")
    if matches.get("evasion_markers"):
        rule_ids.append("evasion:marker")
    if int(matches.get("decoded_segments_count", 0)) > 0:
        rule_ids.append("preprocess:decoded_segments")
    if int(matches.get("joined_obfuscation_sequences_count", 0)) > 0:
        rule_ids.append("preprocess:joined_obfuscation")
    return sorted(set(rule_ids))


def _parse_synergy_pairs(raw_pairs: Any, walls: Sequence[str]) -> List[Tuple[str, str]]:
    if not isinstance(raw_pairs, list) or not raw_pairs:
        return list(DEFAULT_SYNERGY_PAIRS)
    wall_set = set(str(w) for w in walls)
    out: List[Tuple[str, str]] = []
    for item in raw_pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        a = str(item[0]).strip()
        b = str(item[1]).strip()
        if a in wall_set and b in wall_set:
            out.append((a, b))
    return out if out else list(DEFAULT_SYNERGY_PAIRS)


def score_chunks(
    *,
    projector: Any,
    items: Sequence[ContentItem],
    walls: Sequence[str],
    cfg: Mapping[str, Any] | None = None,
) -> ChunkAggregationResult:
    pp_cfg = ChunkPipelineConfig.from_cfg(cfg)
    wall_names = [str(w) for w in walls]
    wall_max = {w: 0.0 for w in wall_names}
    chunk_scores: List[ChunkScore] = []
    projections: List[ProjectionResult] = []
    top_rows: List[Dict[str, Any]] = []
    multiwall_chunks = 0
    pattern_signal_union = set()
    rule_id_union = set()
    triggered_chunk_ids = set()

    for item in items:
        proj = projector.project(item)
        projections.append(proj)
        vec = [float(x) for x in proj.v.tolist()]
        per_wall = {wall_names[i]: vec[i] for i in range(min(4, len(wall_names)))}
        for w, v in per_wall.items():
            if v > wall_max[w]:
                wall_max[w] = v
        active_walls = sorted([w for w, v in per_wall.items() if v >= pp_cfg.wall_trigger_threshold])
        if len(active_walls) >= 2:
            multiwall_chunks += 1
        matches = proj.evidence.matches if isinstance(proj.evidence.matches, dict) else {}
        signals = _extract_pattern_signals(matches)
        rule_ids = _extract_rule_ids(matches)
        for sig in signals:
            pattern_signal_union.add(sig)
        for rid in rule_ids:
            rule_id_union.add(rid)
        if active_walls:
            triggered_chunk_ids.add(str(item.doc_id))
        row = ChunkScore(
            doc_id=str(item.doc_id),
            score_max=max(per_wall.values()) if per_wall else 0.0,
            wall_scores=per_wall,
            active_walls=active_walls,
            pattern_signals=signals,
            matched_rule_ids=rule_ids,
        )
        chunk_scores.append(row)
        top_rows.append(
            {
                "doc_id": row.doc_id,
                "score_max": float(row.score_max),
                "active_walls": list(row.active_walls),
                "pattern_signals": list(row.pattern_signals),
                "rule_ids": list(row.matched_rule_ids),
            }
        )

    top_rows = sorted(top_rows, key=lambda x: (-float(x["score_max"]), str(x["doc_id"])))
    top_rows = top_rows[: max(1, pp_cfg.top_chunks_limit)]
    worst_chunk_score = float(top_rows[0]["score_max"]) if top_rows else 0.0

    pairs = _parse_synergy_pairs((cfg or {}).get("synergy_pairs", []), wall_names)
    pair_hits: List[str] = []
    for a, b in pairs:
        if wall_max.get(a, 0.0) >= pp_cfg.wall_trigger_threshold and wall_max.get(b, 0.0) >= pp_cfg.wall_trigger_threshold:
            pair_hits.append(f"{a}+{b}")
    pair_hits = sorted(set(pair_hits))

    pattern_synergy = (
        (len(pair_hits) * pp_cfg.synergy_pair_bonus)
        + (min(1, multiwall_chunks) * pp_cfg.synergy_multiwall_bonus)
        + (min(3, len(pattern_signal_union)) * pp_cfg.synergy_pattern_bonus)
    )
    pattern_synergy = _clip01(pattern_synergy)

    top1 = float(top_rows[0]["score_max"]) if len(top_rows) >= 1 else 0.0
    top2 = float(top_rows[1]["score_max"]) if len(top_rows) >= 2 else 0.0
    margin = max(0.0, top1 - top2)
    margin_norm = _clip01(margin / max(1e-6, pp_cfg.confidence_margin))
    support_chunks = sum(1 for row in chunk_scores if row.score_max >= pp_cfg.confidence_support_threshold)
    support_norm = _clip01(float(support_chunks) / float(max(1, pp_cfg.confidence_support_chunks)))
    confidence = _clip01((0.50 * _clip01(top1)) + (0.30 * margin_norm) + (0.20 * support_norm))

    weights_sum = max(1e-6, pp_cfg.worst_weight + pp_cfg.synergy_weight + pp_cfg.confidence_weight)
    doc_score = _clip01(
        (
            pp_cfg.worst_weight * worst_chunk_score
            + pp_cfg.synergy_weight * pattern_synergy
            + pp_cfg.confidence_weight * confidence
        )
        / weights_sum
    )

    reasons: List[str] = []
    if worst_chunk_score >= 0.70:
        reasons.append("chunk_worst_high")
    elif worst_chunk_score >= 0.45:
        reasons.append("chunk_worst_medium")
    if pair_hits:
        reasons.append("chunk_synergy_pair")
    if multiwall_chunks > 0:
        reasons.append("chunk_synergy_multiwall")
    if confidence >= 0.70:
        reasons.append("chunk_confidence_high")

    return ChunkAggregationResult(
        chunk_scores=chunk_scores,
        projections=projections,
        wall_max=wall_max,
        worst_chunk_score=worst_chunk_score,
        pattern_synergy=pattern_synergy,
        confidence=confidence,
        doc_score=doc_score,
        pair_hits=pair_hits,
        top_chunks=top_rows,
        rule_ids=sorted(rule_id_union),
        triggered_chunk_ids=sorted(triggered_chunk_ids),
        reasons=sorted(set(reasons)),
    )


def aggregate_semantic_execution_trace(projections: Sequence[ProjectionResult]) -> Dict[str, Any]:
    """Build a self-contained request-scoped semantic trace snapshot from projection evidence.

    This deliberately avoids projector-global ``last request`` state, which is unsafe
    under concurrent API requests.
    """

    traces: List[Dict[str, Any]] = []
    for projection in projections:
        matches = projection.evidence.matches if isinstance(projection.evidence.matches, dict) else {}
        api_match = matches.get("api_perception", {}) if isinstance(matches, dict) else {}
        if not isinstance(api_match, Mapping):
            continue
        trace = api_match.get("execution_trace", {})
        if isinstance(trace, Mapping) and trace:
            row = dict(trace)
        else:
            row = {
                "semantic_input_kind": str(
                    api_match.get(
                        "semantic_input_kind",
                        "image_only" if bool(api_match.get("vision_attempted", False)) else "text_only",
                    )
                ),
                "vision_attempted": bool(api_match.get("vision_attempted", False)),
                "vision_provider_supported": bool(api_match.get("vision_provider_supported", False)),
                "vision_failure_policy": str(api_match.get("vision_failure_policy", "none")),
                "vision_fallback_used": bool(api_match.get("vision_fallback_used", False)),
                "vision_semantic_status": str(api_match.get("vision_semantic_status", "none")),
                "provider_call_count": int(api_match.get("provider_call_count", 0) or 0),
                "retry_count": int(api_match.get("retry_count", 0) or 0),
                "cache_hit": bool(api_match.get("cache_hit_last_request", api_match.get("cache_hit", False))),
                "semantic_latency_ms": api_match.get("semantic_latency_ms"),
                "first_pass_latency_ms": api_match.get("first_pass_latency_ms"),
                "second_pass_latency_ms": api_match.get("second_pass_latency_ms"),
                "second_pass_attempted": bool(api_match.get("second_pass_attempted", False)),
                "second_pass_result": str(api_match.get("second_pass_result", "not_attempted")),
                "token_usage": dict(api_match.get("token_usage", {}) or {}),
                "semantic_status": str(api_match.get("semantic_status", "unknown")),
                "provider": str(api_match.get("provider", "")),
                "provider_id": str(api_match.get("provider_id", "")),
                "provider_capabilities": dict(api_match.get("provider_capabilities", {}) or {}),
                "provider_route": list(api_match.get("provider_route", []) or []),
                "fallback_level": str(api_match.get("fallback_level", "none")),
                "fallback_reason": api_match.get("fallback_reason"),
                "llm_fallback_active": bool(api_match.get("llm_fallback_active", False)),
                "health_state": str(api_match.get("health_state", "healthy")),
                "quota_signal": api_match.get("quota_signal"),
            }
        traces.append(row)

    if not traces:
        return {}
    vision_rows = [row for row in traces if bool(row.get("vision_attempted", False))]
    selected = dict((vision_rows or traces)[-1])
    selected["provider_call_count"] = sum(int(row.get("provider_call_count", 0) or 0) for row in traces)
    selected["retry_count"] = sum(int(row.get("retry_count", 0) or 0) for row in traces)
    selected["cache_hit_last_request"] = bool(selected.get("cache_hit", False))
    selected["llm_fallback_active"] = any(bool(row.get("llm_fallback_active", False)) for row in traces)
    selected["vision_fallback_used"] = any(bool(row.get("vision_fallback_used", False)) for row in vision_rows)

    def _sum_optional(name: str) -> float | None:
        vals = [float(row[name]) for row in traces if isinstance(row.get(name), (int, float))]
        return sum(vals) if vals else None

    selected["semantic_latency_ms"] = _sum_optional("semantic_latency_ms")
    selected["first_pass_latency_ms"] = _sum_optional("first_pass_latency_ms")
    selected["second_pass_latency_ms"] = _sum_optional("second_pass_latency_ms")
    selected["second_pass_attempted"] = any(bool(row.get("second_pass_attempted", False)) for row in traces)

    token_usage: Dict[str, Any] = {}
    for row in traces:
        for key, value in dict(row.get("token_usage", {}) or {}).items():
            if isinstance(value, (int, float)):
                token_usage[str(key)] = float(token_usage.get(str(key), 0.0)) + float(value)
            elif isinstance(value, Mapping):
                nested = dict(token_usage.get(str(key), {}) or {})
                for nkey, nvalue in value.items():
                    if isinstance(nvalue, (int, float)):
                        nested[str(nkey)] = float(nested.get(str(nkey), 0.0)) + float(nvalue)
                    else:
                        nested[str(nkey)] = nvalue
                token_usage[str(key)] = nested
            else:
                token_usage[str(key)] = value
    selected["token_usage"] = token_usage
    selected["trace_source"] = "projection_evidence"
    selected["projection_trace_count"] = len(traces)
    return selected
