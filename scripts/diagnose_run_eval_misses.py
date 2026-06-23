from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem, ProjectionResult, WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2

QUOTED_RE = re.compile(r"""['"]([^'"]{4,80})['"]""")
TOKEN_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "under",
    "when",
    "while",
    "then",
    "than",
    "have",
    "has",
    "may",
    "can",
    "are",
    "was",
    "were",
    "will",
    "would",
    "should",
    "could",
    "their",
    "there",
    "after",
    "before",
    "note",
    "case",
    "prior",
    "latest",
    "context",
}
DEFAULT_PACK_PATH = ROOT / "tests" / "data" / "session_benchmark" / "redteam_cross_source_polish_ab_n24_session_pack.jsonl"
FAMILY_TARGETS: Dict[str, List[str]] = {
    "xsrc_policy_tool": ["policy_evasion", "tool_or_action_abuse"],
    "xsrc_secret_policy": ["secret_exfiltration", "policy_evasion"],
    "xsrc_override_tool": ["override_instructions", "tool_or_action_abuse"],
    "xsrc_override_secret": ["override_instructions", "secret_exfiltration"],
}


@dataclass
class SampleOutcome:
    sample_id: str
    family: str
    targets: List[str]
    detected_targets: List[str]
    non_target_hits: List[str]
    missed_targets: List[str]
    primary_class: str
    flags: List[str]
    raw_scores: Dict[str, float]
    polarity: Dict[str, int]
    tiers: Dict[str, str]
    soft_signals: Dict[str, List[str]]
    text: str


def _wall_idx() -> Dict[str, int]:
    return {w: i for i, w in enumerate(WALLS_V1)}


def _normalize_text(text: str) -> str:
    return str(text or "").strip()


def _load_pack_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _pack_rows_to_samples(rows: Sequence[Mapping[str, Any]], *, families: Sequence[str]) -> List[Dict[str, Any]]:
    family_filter = set(str(x) for x in families if str(x))
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    sample_family: Dict[str, str] = {}

    for row in rows:
        label = str(row.get("label_session", "")).strip().lower()
        if label != "attack":
            continue
        family = str(row.get("family", "")).strip()
        if family_filter and family not in family_filter:
            continue
        source_ref = str(row.get("source_ref", "")).strip()
        sample_id = source_ref.split(":", 1)[0] if source_ref else str(row.get("session_id", "unknown"))
        grouped[sample_id].append(row)
        sample_family[sample_id] = family

    samples: List[Dict[str, Any]] = []
    for sample_id, sample_rows in grouped.items():
        family = sample_family.get(sample_id, "unknown")
        targets = list(FAMILY_TARGETS.get(family, []))
        ordered = sorted(
            sample_rows,
            key=lambda r: (
                str(r.get("session_id", "")),
                int(r.get("turn_id", 0) or 0),
            ),
        )
        text = "\n".join(_normalize_text(str(row.get("text", ""))) for row in ordered if _normalize_text(str(row.get("text", ""))))
        samples.append(
            {
                "id": sample_id,
                "family": family,
                "targets": targets,
                "text": text,
            }
        )
    return sorted(samples, key=lambda item: str(item.get("id", "")))


def _try_generate_samples(families: Sequence[str], seed: int, n_per_family: int) -> Optional[List[Dict[str, Any]]]:
    try:
        from dataclasses import asdict
        from redteam.generator import generate  # type: ignore
    except Exception:
        return None
    samples = [asdict(s) for s in generate(seed=seed, n_per_family=n_per_family)]
    family_filter = set(str(x) for x in families if str(x))
    if family_filter:
        samples = [sample for sample in samples if str(sample.get("family", "")) in family_filter]
    return samples


def _sample_text(sample: Mapping[str, Any]) -> str:
    text = sample.get("text", "")
    if (not text) and sample.get("source_fragments"):
        text = "\n".join(
            str((frag or {}).get("text", ""))
            for frag in (sample.get("source_fragments") or [])
        )
    return str(text)


def _rule_tier_meta(proj: ProjectionResult) -> Mapping[str, Any]:
    evidence = getattr(proj, "evidence", None)
    matches = getattr(evidence, "matches", {}) if evidence is not None else {}
    meta = matches.get("pi0_rule_tier") if isinstance(matches, Mapping) else {}
    return meta if isinstance(meta, Mapping) else {}


def classify_sample_outcome(sample: Mapping[str, Any], proj: ProjectionResult) -> SampleOutcome:
    idx = _wall_idx()
    targets = [str(w) for w in list(sample.get("targets", []) or []) if str(w) in idx]
    v_list = [float(x) for x in list(proj.v)]
    evidence = proj.evidence
    raw_list = [float(x) for x in list(getattr(evidence, "debug_scores_raw", []) or [])]
    polarity_list = [int(x) for x in list(getattr(evidence, "polarity", []) or [])]
    tier_meta = _rule_tier_meta(proj)

    detected_targets = [wall for wall in targets if idx.get(wall, -1) < len(v_list) and v_list[idx[wall]] > 0.0]
    missed_targets = [wall for wall in targets if wall not in detected_targets]
    non_target_hits = [
        wall
        for wall in WALLS_V1
        if wall not in targets and idx.get(wall, -1) < len(v_list) and v_list[idx[wall]] > 0.0
    ]

    raw_scores = {wall: (raw_list[idx[wall]] if idx[wall] < len(raw_list) else 0.0) for wall in WALLS_V1}
    polarity = {wall: (polarity_list[idx[wall]] if idx[wall] < len(polarity_list) else 0) for wall in WALLS_V1}
    tiers = {wall: str((tier_meta.get(wall) or {}).get("tier", "none")) for wall in WALLS_V1}
    soft_signals = {
        wall: [str(x) for x in list(((tier_meta.get(wall) or {}).get("soft_signals") or []))]
        for wall in WALLS_V1
    }

    flags: List[str] = []
    if missed_targets:
        if all(raw_scores[wall] > 0.0 and polarity[wall] <= 0 for wall in missed_targets):
            flags.extend(["raw_positive_final_zero", "polarity_block"])
        elif any(raw_scores[wall] > 0.0 and polarity[wall] <= 0 for wall in missed_targets):
            flags.append("partial_polarity_block")
        if any(
            raw_scores[wall] > 0.0
            and polarity[wall] <= 0
            and "semantic_boost" in soft_signals.get(wall, [])
            for wall in missed_targets
        ):
            flags.append("semantic_boost_only")
        if non_target_hits:
            flags.append("misrouting_non_target_hit")

    if not missed_targets:
        primary_class = "full_hit"
    elif detected_targets:
        primary_class = "partial_target_hit"
    elif non_target_hits:
        primary_class = "wrong_wall_zero_target"
    else:
        primary_class = "zero_detection"

    return SampleOutcome(
        sample_id=str(sample.get("id", "sample")),
        family=str(sample.get("family", "unknown")),
        targets=targets,
        detected_targets=detected_targets,
        non_target_hits=non_target_hits,
        missed_targets=missed_targets,
        primary_class=primary_class,
        flags=flags,
        raw_scores=raw_scores,
        polarity=polarity,
        tiers=tiers,
        soft_signals=soft_signals,
        text=_sample_text(sample),
    )


def evaluate_samples_detailed(samples: Iterable[Mapping[str, Any]], projector: Pi0IntentAwareV2) -> List[SampleOutcome]:
    outcomes: List[SampleOutcome] = []
    for sample in samples:
        item = ContentItem(
            doc_id=str(sample.get("id", "sample")),
            source_id="diagnostic:sample",
            source_type="other",
            trust="untrusted",
            text=_sample_text(sample),
        )
        proj = projector.project(item)
        outcomes.append(classify_sample_outcome(sample, proj))
    return outcomes


def _top_phrase_hints(outcomes: Sequence[SampleOutcome], top_n: int) -> List[Dict[str, Any]]:
    quoted = Counter()
    ngrams = Counter()
    for outcome in outcomes:
        text = outcome.text.lower()
        for match in QUOTED_RE.findall(text):
            phrase = " ".join(match.split())
            if phrase:
                quoted[phrase] += 1
        tokens = [t for t in TOKEN_RE.findall(text) if len(t) >= 4 and t not in STOPWORDS]
        for n in (2, 3, 4):
            for i in range(0, max(0, len(tokens) - n + 1)):
                phrase = " ".join(tokens[i : i + n])
                if len(set(tokens[i : i + n])) == 1:
                    continue
                ngrams[phrase] += 1
    ranked: List[Dict[str, Any]] = []
    seen = set()
    for phrase, count in quoted.most_common(top_n):
        ranked.append({"phrase": phrase, "count": int(count), "kind": "quoted"})
        seen.add(phrase)
    for phrase, count in ngrams.most_common(top_n * 3):
        if phrase in seen:
            continue
        ranked.append({"phrase": phrase, "count": int(count), "kind": "ngram"})
        seen.add(phrase)
        if len(ranked) >= top_n:
            break
    return ranked[:top_n]


def _phrases_for_outcome(outcome: SampleOutcome) -> List[str]:
    phrases: List[str] = []
    text = outcome.text.lower()
    for match in QUOTED_RE.findall(text):
        phrase = " ".join(match.split())
        if phrase:
            phrases.append(phrase)
    if phrases:
        return phrases
    tokens = [t for t in TOKEN_RE.findall(text) if len(t) >= 4 and t not in STOPWORDS]
    for n in (4, 3, 2):
        for i in range(0, max(0, len(tokens) - n + 1)):
            phrase = " ".join(tokens[i : i + n])
            if len(set(tokens[i : i + n])) == 1:
                continue
            phrases.append(phrase)
    return phrases


def _bool_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _wall_cluster_summary(outcomes: Sequence[SampleOutcome], wall: str, top_n: int) -> Dict[str, Any]:
    phrase_counter = Counter()
    raw_positive_count = 0
    polarity_zero_count = 0
    misrouted_count = 0
    semantic_boost_only_count = 0
    tier_counter = Counter()
    exemplar_rows: List[Dict[str, Any]] = []
    for outcome in outcomes:
        raw_positive = float(outcome.raw_scores.get(wall, 0.0)) > 0.0
        polarity_zero = int(outcome.polarity.get(wall, 0)) == 0
        misrouted = bool(outcome.non_target_hits)
        semantic_boost_only = "semantic_boost" in list(outcome.soft_signals.get(wall, []))
        tier_counter[str(outcome.tiers.get(wall, "none"))] += 1
        if raw_positive:
            raw_positive_count += 1
        if polarity_zero:
            polarity_zero_count += 1
        if misrouted:
            misrouted_count += 1
        if semantic_boost_only:
            semantic_boost_only_count += 1
        for phrase in _phrases_for_outcome(outcome):
            phrase_counter[phrase] += 1
    for outcome in outcomes[:top_n]:
        exemplar_rows.append(
            {
                "sample_id": outcome.sample_id,
                "primary_class": outcome.primary_class,
                "raw_score": float(outcome.raw_scores.get(wall, 0.0)),
                "polarity": int(outcome.polarity.get(wall, 0)),
                "tier": str(outcome.tiers.get(wall, "none")),
                "misrouted_to": list(outcome.non_target_hits),
                "text": outcome.text,
            }
        )
    return {
        "count": len(outcomes),
        "raw_positive_count": raw_positive_count,
        "raw_positive_rate": _bool_rate(raw_positive_count, len(outcomes)),
        "polarity_zero_count": polarity_zero_count,
        "polarity_zero_rate": _bool_rate(polarity_zero_count, len(outcomes)),
        "misrouted_count": misrouted_count,
        "misrouted_rate": _bool_rate(misrouted_count, len(outcomes)),
        "semantic_boost_only_count": semantic_boost_only_count,
        "semantic_boost_only_rate": _bool_rate(semantic_boost_only_count, len(outcomes)),
        "tier_counts": dict(tier_counter),
        "top_phrase_clusters": [{"phrase": p, "count": int(c)} for p, c in phrase_counter.most_common(top_n)],
        "examples": exemplar_rows,
    }


def _family_summary(outcomes: Sequence[SampleOutcome], top_n: int) -> Dict[str, Any]:
    primary = Counter()
    flags = Counter()
    missed_wall = Counter()
    misrouted_wall = Counter()
    tier_pairs = Counter()
    for outcome in outcomes:
        primary[outcome.primary_class] += 1
        for flag in outcome.flags:
            flags[flag] += 1
        for wall in outcome.missed_targets:
            missed_wall[wall] += 1
            tier_pairs[f"{wall}:{outcome.tiers.get(wall, 'none')}"] += 1
        for wall in outcome.non_target_hits:
            misrouted_wall[wall] += 1
    exemplars = []
    for outcome in outcomes[:top_n]:
        exemplars.append(
            {
                "sample_id": outcome.sample_id,
                "primary_class": outcome.primary_class,
                "flags": list(outcome.flags),
                "targets": list(outcome.targets),
                "detected_targets": list(outcome.detected_targets),
                "missed_targets": list(outcome.missed_targets),
                "non_target_hits": list(outcome.non_target_hits),
                "tiers": dict(outcome.tiers),
                "raw_scores": dict(outcome.raw_scores),
                "polarity": dict(outcome.polarity),
                "text": outcome.text,
            }
        )
    return {
        "count": len(outcomes),
        "primary_class_counts": dict(primary),
        "flag_counts": dict(flags),
        "missed_wall_counts": dict(missed_wall),
        "misrouted_wall_counts": dict(misrouted_wall),
        "missed_wall_tiers": dict(tier_pairs),
        "top_phrase_hints": _top_phrase_hints(outcomes, top_n=top_n),
        "examples": exemplars,
    }


def _build_wall_miss_table(outcomes: Sequence[SampleOutcome], top_n: int) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str], List[SampleOutcome]] = defaultdict(list)
    for outcome in outcomes:
        for wall in outcome.missed_targets:
            grouped[(outcome.family, wall)].append(outcome)
    rows: List[Dict[str, Any]] = []
    for (family, wall), wall_outcomes in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        rows.append({"family": family, "missed_wall": wall, **_wall_cluster_summary(wall_outcomes, wall, top_n=top_n)})
    return rows


def _build_phrase_wall_priority_table(outcomes: Sequence[SampleOutcome], top_n: int) -> List[Dict[str, Any]]:
    grouped: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for outcome in outcomes:
        for wall in outcome.missed_targets:
            phrases = _phrases_for_outcome(outcome)
            for phrase in phrases[: max(1, top_n)]:
                key = (outcome.family, wall, phrase)
                bucket = grouped.setdefault(
                    key,
                    {
                        "family": outcome.family,
                        "missed_wall": wall,
                        "phrase_cluster": phrase,
                        "count": 0,
                        "raw_positive_count": 0,
                        "polarity_zero_count": 0,
                        "raw_positive_polarity_zero_count": 0,
                        "misrouted_count": 0,
                        "sample_ids": [],
                    },
                )
                bucket["count"] += 1
                raw_positive = float(outcome.raw_scores.get(wall, 0.0)) > 0.0
                polarity_zero = int(outcome.polarity.get(wall, 0)) == 0
                if raw_positive:
                    bucket["raw_positive_count"] += 1
                if polarity_zero:
                    bucket["polarity_zero_count"] += 1
                if raw_positive and polarity_zero:
                    bucket["raw_positive_polarity_zero_count"] += 1
                if outcome.non_target_hits:
                    bucket["misrouted_count"] += 1
                if len(bucket["sample_ids"]) < top_n:
                    bucket["sample_ids"].append(outcome.sample_id)

    rows: List[Dict[str, Any]] = []
    for bucket in grouped.values():
        count = int(bucket["count"])
        raw_positive_count = int(bucket["raw_positive_count"])
        polarity_zero_count = int(bucket["polarity_zero_count"])
        raw_positive_polarity_zero_count = int(bucket["raw_positive_polarity_zero_count"])
        misrouted_count = int(bucket["misrouted_count"])
        rows.append(
            {
                **bucket,
                "raw_positive_rate": _bool_rate(raw_positive_count, count),
                "polarity_zero_rate": _bool_rate(polarity_zero_count, count),
                "raw_positive_polarity_zero_rate": _bool_rate(raw_positive_polarity_zero_count, count),
                "misrouted_rate": _bool_rate(misrouted_count, count),
            }
        )

    rows.sort(
        key=lambda row: (
            -int(row["raw_positive_polarity_zero_count"]),
            -int(row["count"]),
            str(row["family"]),
            str(row["missed_wall"]),
            str(row["phrase_cluster"]),
        )
    )
    return rows[: max(top_n * 4, top_n)]


def build_diagnostic_report(outcomes: Sequence[SampleOutcome], families: Sequence[str], top_n: int, *, data_source: str) -> Dict[str, Any]:
    family_filter = set(families)
    filtered = [outcome for outcome in outcomes if (not family_filter or outcome.family in family_filter)]
    misses = [outcome for outcome in filtered if outcome.primary_class != "full_hit"]
    by_family: Dict[str, List[SampleOutcome]] = defaultdict(list)
    for outcome in misses:
        by_family[outcome.family].append(outcome)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_source": data_source,
        "families": list(families),
        "overall": {
            "total_outcomes": len(filtered),
            "miss_count": len(misses),
            "primary_class_counts": dict(Counter(outcome.primary_class for outcome in misses)),
            "flag_counts": dict(Counter(flag for outcome in misses for flag in outcome.flags)),
        },
        "wall_miss_table": _build_wall_miss_table(misses, top_n=top_n),
        "phrase_wall_priority_table": _build_phrase_wall_priority_table(misses, top_n=top_n),
        "families_report": {
            family: _family_summary(family_outcomes, top_n=top_n)
            for family, family_outcomes in sorted(by_family.items())
        },
    }


def _load_samples(families: Sequence[str], seed: int, n_per_family: int, pack_path: Path) -> tuple[List[Dict[str, Any]], str]:
    generated = _try_generate_samples(families=families, seed=seed, n_per_family=n_per_family)
    if generated is not None:
        return generated, "redteam_generator"
    pack_rows = _load_pack_rows(pack_path)
    return _pack_rows_to_samples(pack_rows, families=families), f"session_pack:{pack_path.as_posix()}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose run_eval misses by family and missed-wall taxonomy.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-per-family", type=int, default=200)
    parser.add_argument("--pack-path", default=str(DEFAULT_PACK_PATH))
    parser.add_argument("--family", action="append", default=[], help="Restrict report to one or more families; repeatable.")
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    families = list(args.family)
    snapshot = load_resolved_config(profile=str(args.profile))
    projector = Pi0IntentAwareV2(snapshot.resolved)
    samples, data_source = _load_samples(
        families=families,
        seed=int(args.seed),
        n_per_family=int(args.n_per_family),
        pack_path=Path(str(args.pack_path)),
    )
    outcomes = evaluate_samples_detailed(samples, projector)
    report = build_diagnostic_report(outcomes, families=families, top_n=int(args.top_n), data_source=data_source)
    payload = json.dumps(report, ensure_ascii=True, indent=2)
    if args.json_out:
        Path(args.json_out).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
