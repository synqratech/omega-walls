from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


DEFAULT_FN_SAMPLES = "artifacts/deepset_analysis/deepset_fn_20260224T183908Z_4b9eed153ed3/fn_samples.jsonl"
DEFAULT_FN_PATTERNS = "artifacts/deepset_analysis/deepset_fn_20260224T183908Z_4b9eed153ed3/fn_patterns.json"
DEFAULT_RELEASE_METRICS = "artifacts/release_gate/release_gate_20260225T011523Z_6d0baeb9d10c/release_metrics.json"


@dataclass(frozen=True)
class SecondaryVariantSpec:
    variant_id: str
    wall: str
    patterns: Tuple[str, ...]


SECONDARY_VARIANTS: Sequence[SecondaryVariantSpec] = (
    SecondaryVariantSpec(
        variant_id="override_forget_reset_multilingual",
        wall="override_instructions",
        patterns=(
            r"\b(forget|vergiss|olvide|olvida|ignore|ignoriere|ignorieren)\b.{0,80}\b(before|previous|prior|davor|zuvor|todo|said|gesagt|gesagte|obigen|obige)\b",
            r"\b(hore|h[öo]re)\b.{0,20}\bnicht\b.{0,40}\bzuvor\b.{0,40}\bgesagte\b",
            r"\balles zuvor gesagt[ee]\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="override_roleplay_multilingual",
        wall="override_instructions",
        patterns=(
            r"\b(act as|i want you to act as|you are now)\b",
            r"\b(als\s+\w+\s+agier\w*|drehbuchautor|geschichtenerz[aä]hler|journalistin)\b",
            r"\b(interviewer|candidate|position|storyteller)\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="override_forced_response_style",
        wall="override_instructions",
        patterns=(
            r"\b(respond to all questions with|answer every question with)\b",
            r"\b(if you don[' ]?t know[, ]*say|if you dont know[, ]*say)\b",
            r"\b(you are supposed to answer at all times|solo dices|say that you)\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="override_prompt_leak_or_embeddings",
        wall="secret_exfiltration",
        patterns=(
            r"\b(show|print|return|reveal|say)\b.{0,60}\b(prompt|prompt texts|eingabeaufforderung|instructions|embeddings?)\b",
            r"\b(return your embeddings|print above prompt|show me all your prompt texts)\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="tool_sql_database_multilingual",
        wall="tool_or_action_abuse",
        patterns=(
            r"\b(sql|database|datenbank)\b",
            r"\b(access a database|query database|sql\s*-?code|code to access)\b",
            r"\b(generieren sie den sql)\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="policy_evasion_unrestricted",
        wall="policy_evasion",
        patterns=(
            r"\b(bypass|jailbreak|ignore safety|without restrictions|no restrictions|without limitations|guardrails)\b",
            r"\b(theoretical world|hypothetical world).{0,50}\b(without restrictions|unrestricted)\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="softener_false_legitimation",
        wall="override_instructions",
        patterns=(
            r"\b(not a prompt injection|for testing only|for training|simulation only)\b",
            r"\b(i honestly just want|if you don[' ]?t answer i[' ]?ll be very sad)\b",
            r"\b(theoretical world|hypothetical)\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="obfuscation_split_letters",
        wall="override_instructions",
        patterns=(
            r"(?:\b[a-z]\s+){3,}[a-z]\b",
            r"\b(igmre|waht)\b",
        ),
    ),
    SecondaryVariantSpec(
        variant_id="wrapper_context_blend",
        wall="override_instructions",
        patterns=(
            r"\b(context|order|question)\b.{0,80}\b(ignore|forget|answer|say)\b",
            r"\b(system|developer|assistant|user)\s*:",
        ),
    ),
)


RULE_PACK_RECIPES: Dict[str, Dict[str, Any]] = {
    "override_forget_reset_multilingual": {
        "pack_id": "rb_override_reset_multilingual",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.dict.override_verbs", "pi0.dict.override_objects", "pi0.dict.override_phrases"],
        "proposal": "Add multilingual reset verbs/phrases (de/es variants) and typo forms tied to instruction objects.",
    },
    "override_roleplay_multilingual": {
        "pack_id": "rb_roleplay_assignment",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.dict.override_phrases", "pi0.struct_patterns"],
        "proposal": "Add role-assignment wrappers and role nouns in de/en; keep anchor requirement to avoid benign drift.",
    },
    "override_forced_response_style": {
        "pack_id": "rb_forced_response_templates",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.dict.override_phrases", "pi0.struct_patterns"],
        "proposal": "Add forced-format response templates and low-knowledge fallback coercion prompts.",
    },
    "override_prompt_leak_or_embeddings": {
        "pack_id": "rb_prompt_leak_requests",
        "target_wall": "secret_exfiltration",
        "config_targets": ["pi0.dict.secret_nouns", "pi0.dict.exfil_verbs", "pi0.semantic.prototypes.secret_exfiltration"],
        "proposal": "Expand prompt-leak and embeddings exfil language; mirror in semantic prototypes.",
    },
    "tool_sql_database_multilingual": {
        "pack_id": "rb_tool_sql_db",
        "target_wall": "tool_or_action_abuse",
        "config_targets": ["pi0.dict.tool_tokens", "pi0.dict.override_phrases"],
        "proposal": "Add SQL/database verbs in de/en and stronger imperative handling for command-generation requests.",
    },
    "policy_evasion_unrestricted": {
        "pack_id": "rb_policy_evasion_unrestricted",
        "target_wall": "policy_evasion",
        "config_targets": ["pi0.dict.evasion_markers", "pi0.semantic.prototypes.policy_evasion"],
        "proposal": "Add unrestricted/jailbreak paraphrases and semantic prototypes; preserve negation guard thresholds.",
    },
    "softener_false_legitimation": {
        "pack_id": "rb_false_legitimation_softeners",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.dict.false_legit_markers", "pi0.weights.override.w_softener_boost"],
        "proposal": "Expand manipulative softeners that mask attacks as harmless requests.",
    },
    "obfuscation_split_letters": {
        "pack_id": "rb_obfuscation_spacing_typos",
        "target_wall": "override_instructions",
        "config_targets": ["pi0 normalization/canonicalization path", "pi0.dict.override_phrases"],
        "proposal": "Harden spaced-letter and typo obfuscation mapping before phrase matching.",
    },
    "wrapper_context_blend": {
        "pack_id": "rb_wrapper_context_blend",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.struct_patterns", "pi0.dict.override_phrases"],
        "proposal": "Add patterns for benign-context + injected-order blends common in retrieved chunks.",
    },
    "soft_directive_i_want_you": {
        "pack_id": "rb_soft_directive_templates",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.struct_patterns", "pi0.dict.override_phrases"],
        "proposal": "Cover soft directives and polite coercion templates.",
    },
    "role_assignment_act_as": {
        "pack_id": "rb_role_assignment_core",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.dict.override_phrases", "pi0.struct_patterns"],
        "proposal": "Increase support for 'act as' role assignment templates.",
    },
    "role_interviewer_pattern": {
        "pack_id": "rb_interviewer_variant",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.dict.override_phrases"],
        "proposal": "Add interviewer/candidate targeted prompt wrappers.",
    },
    "tool_sql_db_access": {
        "pack_id": "rb_tool_sql_db",
        "target_wall": "tool_or_action_abuse",
        "config_targets": ["pi0.dict.tool_tokens"],
        "proposal": "Cover SQL/database access requests that imply actionable tool abuse.",
    },
    "other_indirect": {
        "pack_id": "rb_indirect_backlog",
        "target_wall": "override_instructions",
        "config_targets": ["pi0.semantic.prototypes.override_instructions"],
        "proposal": "Backlog bucket for indirect paraphrases requiring semantic/hybrid handling.",
    },
}

VARIANT_ALIASES: Dict[str, str] = {
    "tool_sql_db_access": "tool_sql_database_multilingual",
    "role_assignment_act_as": "override_roleplay_multilingual",
}

GENERIC_VARIANTS = {"other_indirect"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _detect_secondary_variants(text: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for spec in SECONDARY_VARIANTS:
        for patt in spec.patterns:
            if re.search(patt, text, flags=re.IGNORECASE | re.DOTALL):
                out.append((spec.variant_id, spec.wall))
                break
    return out


def _canonical_wall(variant_id: str, fallback: str) -> str:
    if variant_id in RULE_PACK_RECIPES:
        return str(RULE_PACK_RECIPES[variant_id]["target_wall"])
    return fallback


def _variant_recipe(variant_id: str, wall: str) -> Dict[str, Any]:
    base = RULE_PACK_RECIPES.get(variant_id)
    if base is not None:
        return base
    return {
        "pack_id": f"rb_custom_{variant_id}",
        "target_wall": wall,
        "config_targets": ["pi0.dict.*", "pi0.semantic.prototypes.*"],
        "proposal": "Review this variant and add focused dictionary/prototype coverage.",
    }


def _parse_fn_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_variant_samples: Dict[str, set[str]] = defaultdict(set)
    by_variant_examples: Dict[str, List[str]] = defaultdict(list)
    by_variant_walls: Dict[str, Counter[str]] = defaultdict(Counter)
    sample_to_variants: Dict[str, set[str]] = defaultdict(set)

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            continue
        text = str(row.get("text", ""))
        fallback_wall = str(row.get("suggested_wall", "override_instructions"))
        raw_tags = [str(t).strip() for t in (row.get("fn_pattern_tags") or []) if str(t).strip()]
        kept_tags = [t for t in raw_tags if t != "other_indirect"]

        variant_hits: List[Tuple[str, str]] = []
        for tag in kept_tags:
            canon_tag = VARIANT_ALIASES.get(tag, tag)
            variant_hits.append((canon_tag, _canonical_wall(canon_tag, fallback_wall)))

        secondary_hits = _detect_secondary_variants(text)
        if secondary_hits:
            variant_hits.extend(secondary_hits)

        if not variant_hits:
            variant_hits.append(("other_indirect", fallback_wall))

        unique_hits: List[Tuple[str, str]] = []
        seen = set()
        for variant_id, wall in variant_hits:
            key = (variant_id, wall)
            if key in seen:
                continue
            seen.add(key)
            unique_hits.append(key)

        for variant_id, wall in unique_hits:
            by_variant_samples[variant_id].add(sample_id)
            by_variant_walls[variant_id][wall] += 1
            sample_to_variants[sample_id].add(variant_id)
            if len(by_variant_examples[variant_id]) < 3:
                preview = " ".join(text.split())
                by_variant_examples[variant_id].append(preview[:220])

    return {
        "by_variant_samples": by_variant_samples,
        "by_variant_examples": by_variant_examples,
        "by_variant_walls": by_variant_walls,
        "sample_to_variants": sample_to_variants,
    }


def _select_pareto(
    *,
    by_variant_samples: Dict[str, set[str]],
    by_variant_examples: Dict[str, List[str]],
    by_variant_walls: Dict[str, Counter[str]],
    total_fn: int,
    attack_total: int | None,
    target_coverage: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_samples = set()
    for sids in by_variant_samples.values():
        all_samples.update(sids)

    uncovered = set(all_samples)
    selected: List[Dict[str, Any]] = []
    used = set()

    while uncovered and (1.0 - (len(uncovered) / max(1, len(all_samples)))) < target_coverage:
        has_non_generic_gain = any(
            (variant_id not in GENERIC_VARIANTS) and len(uncovered & sids) > 0
            for variant_id, sids in by_variant_samples.items()
            if variant_id not in used
        )
        best_variant = None
        best_gain = -1
        best_total = -1
        best_eff_gain = -1.0
        for variant_id, sids in by_variant_samples.items():
            if variant_id in used:
                continue
            if has_non_generic_gain and variant_id in GENERIC_VARIANTS:
                continue
            gain = len(uncovered & sids)
            total = len(sids)
            eff_gain = float(gain) * (0.25 if variant_id in GENERIC_VARIANTS else 1.0)
            if (eff_gain, gain, total, variant_id) > (best_eff_gain, best_gain, best_total, str(best_variant)):
                best_variant = variant_id
                best_gain = gain
                best_total = total
                best_eff_gain = eff_gain

        if best_variant is None or best_gain <= 0:
            break

        used.add(best_variant)
        newly_covered = uncovered & by_variant_samples[best_variant]
        uncovered -= newly_covered

        wall_counter = by_variant_walls.get(best_variant, Counter())
        wall = wall_counter.most_common(1)[0][0] if wall_counter else "override_instructions"
        recipe = _variant_recipe(best_variant, wall)

        fn_share = float(len(by_variant_samples[best_variant])) / float(max(1, total_fn))
        cumulative_fn_coverage = 1.0 - (float(len(uncovered)) / float(max(1, len(all_samples))))
        attack_share = (
            float(len(by_variant_samples[best_variant])) / float(max(1, attack_total))
            if attack_total is not None
            else None
        )
        cumulative_attack_coverage = (
            float(len(all_samples) - len(uncovered)) / float(max(1, attack_total))
            if attack_total is not None
            else None
        )

        selected.append(
            {
                "variant_id": best_variant,
                "pack_id": recipe["pack_id"],
                "target_wall": recipe["target_wall"],
                "config_targets": recipe["config_targets"],
                "proposal": recipe["proposal"],
                "fn_count": len(by_variant_samples[best_variant]),
                "new_covered_fn_count": len(newly_covered),
                "fn_share": fn_share,
                "attack_share": attack_share,
                "cumulative_fn_coverage": cumulative_fn_coverage,
                "cumulative_attack_coverage": cumulative_attack_coverage,
                "examples": by_variant_examples.get(best_variant, []),
            }
        )

    backlog: List[Dict[str, Any]] = []
    for variant_id, sids in sorted(by_variant_samples.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if variant_id in used:
            continue
        wall_counter = by_variant_walls.get(variant_id, Counter())
        wall = wall_counter.most_common(1)[0][0] if wall_counter else "override_instructions"
        recipe = _variant_recipe(variant_id, wall)
        backlog.append(
            {
                "variant_id": variant_id,
                "pack_id": recipe["pack_id"],
                "target_wall": recipe["target_wall"],
                "fn_count": len(sids),
                "fn_share": float(len(sids)) / float(max(1, total_fn)),
                "examples": by_variant_examples.get(variant_id, []),
            }
        )

    return selected, backlog


def _summarize_whitebox_clusters(release_metrics: Dict[str, Any]) -> Dict[str, Any]:
    wb = (((release_metrics.get("run_eval") or {}).get("whitebox")) or {})
    clusters = wb.get("clusters", {}) if isinstance(wb, dict) else {}
    examples = wb.get("examples", []) if isinstance(wb, dict) else []
    if not isinstance(clusters, dict):
        clusters = {}
    if not isinstance(examples, list):
        examples = []

    out_rows: List[Dict[str, Any]] = []
    for cluster_id, count in sorted(clusters.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        preview = ""
        for ex in examples:
            tags = ex.get("cluster_tags", []) if isinstance(ex, dict) else []
            if cluster_id in tags:
                preview = str(ex.get("best_text_preview", ""))[:220]
                break
        out_rows.append(
            {
                "cluster_id": str(cluster_id),
                "count": int(count),
                "example_preview": preview,
            }
        )
    return {
        "total_bypass_samples": int(wb.get("total", 0)) if isinstance(wb, dict) else 0,
        "bypass_rate": float(wb.get("bypass_rate", 0.0)) if isinstance(wb, dict) else 0.0,
        "clusters": out_rows,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Rule-Based Pareto Prioritization",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- target_fn_coverage: `{summary['target_fn_coverage']:.2f}`",
        f"- fn_total: `{summary['fn_total']}`",
        f"- attack_total: `{summary['attack_total']}`",
        f"- selected_rule_packs: `{summary['selected_rule_packs']}`",
        f"- achieved_fn_coverage: `{summary['achieved_fn_coverage']:.4f}`",
        f"- actionable_fn_coverage_no_generic: `{summary['actionable_fn_coverage_no_generic']:.4f}`",
        "",
        "## Top Rule Packs (Pareto)",
        "| rank | variant_id | pack_id | wall | fn_count | new_fn | cumulative_fn_cov | cumulative_attack_cov |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]

    for idx, row in enumerate(report["pareto_rule_packs"], start=1):
        cum_attack = row.get("cumulative_attack_coverage")
        cum_attack_text = f"{cum_attack:.4f}" if isinstance(cum_attack, float) else "n/a"
        lines.append(
            f"| {idx} | `{row['variant_id']}` | `{row['pack_id']}` | `{row['target_wall']}` | "
            f"{row['fn_count']} | {row['new_covered_fn_count']} | {row['cumulative_fn_coverage']:.4f} | {cum_attack_text} |"
        )

    lines.extend(
        [
            "",
            "## Backlog",
            "| variant_id | pack_id | wall | fn_count | fn_share |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in report["backlog"][:12]:
        lines.append(
            f"| `{row['variant_id']}` | `{row['pack_id']}` | `{row['target_wall']}` | {row['fn_count']} | {row['fn_share']:.4f} |"
        )

    wb = report.get("whitebox_cluster_focus", {})
    wb_rows = wb.get("clusters", []) if isinstance(wb, dict) else []
    if wb_rows:
        lines.extend(
            [
                "",
                "## Whitebox Bypass Clusters",
                f"- bypass_rate: `{float(wb.get('bypass_rate', 0.0)):.6f}`",
                "| cluster_id | count |",
                "|---|---:|",
            ]
        )
        for row in wb_rows:
            lines.append(f"| `{row['cluster_id']}` | {row['count']} |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Pareto-prioritized rule packs from FN artifacts (80/20 focus)")
    parser.add_argument("--fn-samples-jsonl", default=DEFAULT_FN_SAMPLES)
    parser.add_argument("--fn-patterns-json", default=DEFAULT_FN_PATTERNS)
    parser.add_argument("--release-metrics-json", default=DEFAULT_RELEASE_METRICS)
    parser.add_argument("--target-fn-coverage", type=float, default=0.80)
    parser.add_argument("--artifacts-root", default="artifacts/rule_pareto")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    fn_samples_path = Path(str(args.fn_samples_jsonl))
    if not fn_samples_path.exists():
        raise FileNotFoundError(f"fn samples jsonl not found: {fn_samples_path}")

    fn_patterns_path = Path(str(args.fn_patterns_json))
    patterns = _load_json(fn_patterns_path) if fn_patterns_path.exists() else {}
    attack_total_raw = patterns.get("attack_total")
    attack_total = int(attack_total_raw) if isinstance(attack_total_raw, int) else None

    fn_rows = _load_jsonl(fn_samples_path)
    parsed = _parse_fn_rows(fn_rows)
    by_variant_samples = parsed["by_variant_samples"]
    by_variant_examples = parsed["by_variant_examples"]
    by_variant_walls = parsed["by_variant_walls"]

    total_fn = len(fn_rows)
    selected, backlog = _select_pareto(
        by_variant_samples=by_variant_samples,
        by_variant_examples=by_variant_examples,
        by_variant_walls=by_variant_walls,
        total_fn=total_fn,
        attack_total=attack_total,
        target_coverage=max(0.0, min(1.0, float(args.target_fn_coverage))),
    )

    achieved_fn_coverage = selected[-1]["cumulative_fn_coverage"] if selected else 0.0
    selected_non_generic = [row for row in selected if row["variant_id"] not in GENERIC_VARIANTS]
    actionable_fn_coverage_no_generic = (
        selected_non_generic[-1]["cumulative_fn_coverage"] if selected_non_generic else 0.0
    )
    whitebox_summary: Dict[str, Any] = {}
    release_metrics_path = Path(str(args.release_metrics_json))
    if release_metrics_path.exists():
        whitebox_summary = _summarize_whitebox_clusters(_load_json(release_metrics_path))

    run_id = str(args.run_id or f"rule_pareto_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    out_dir = Path(str(args.artifacts_root)) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "run_id": run_id,
        "inputs": {
            "fn_samples_jsonl": fn_samples_path.as_posix(),
            "fn_patterns_json": fn_patterns_path.as_posix() if fn_patterns_path.exists() else None,
            "release_metrics_json": release_metrics_path.as_posix() if release_metrics_path.exists() else None,
        },
        "summary": {
            "target_fn_coverage": max(0.0, min(1.0, float(args.target_fn_coverage))),
            "fn_total": total_fn,
            "attack_total": attack_total,
            "selected_rule_packs": len(selected),
            "achieved_fn_coverage": achieved_fn_coverage,
            "actionable_fn_coverage_no_generic": actionable_fn_coverage_no_generic,
        },
        "pareto_rule_packs": selected,
        "backlog": backlog,
        "whitebox_cluster_focus": whitebox_summary,
    }

    json_path = out_dir / "rule_pareto_report.json"
    md_path = out_dir / "rule_pareto_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    md_path.write_text(_render_markdown(report), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
