"""False-negative analysis utilities for deepset benchmark."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

from omega.core.omega_core import OmegaCoreV1
from omega.eval.deepset_adapter import DeepsetSample, build_deepset_samples
from omega.interfaces.contracts_v1 import ContentItem, OmegaState, WALLS_V1
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2


PATTERN_SPECS: Sequence[Dict[str, str]] = (
    {
        "id": "override_ignore_previous",
        "wall": "override_instructions",
        "regex": r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|above|earlier)\b.{0,40}\b(instruction|task|rule)\b",
    },
    {
        "id": "override_new_task_focus",
        "wall": "override_instructions",
        "regex": r"\b(now\s+focus|new\s+task|your\s+new\s+task)\b",
    },
    {
        "id": "role_assignment_act_as",
        "wall": "override_instructions",
        "regex": r"\b(act as|you are now|i want you to act as|ich m[o\u00f6]chte dass sie als|quiero que actues como|actua como)\b",
    },
    {
        "id": "role_interviewer_pattern",
        "wall": "override_instructions",
        "regex": r"\b(interviewer|candidate|position)\b",
    },
    {
        "id": "prompt_leak_request",
        "wall": "secret_exfiltration",
        "regex": r"\b(show|print|reveal|display|return|drucken|drucke|zeige|ausgeben)\b.{0,60}\b(prompt|system prompt|prompt text|prompt texts|eingabeaufforderung|embeddings?)\b",
    },
    {
        "id": "spellcheck_print_above_prompt",
        "wall": "secret_exfiltration",
        "regex": r"\b(spell[\s-]?check|rechtschreibung)\b.{0,80}\b(print|show|drucken|drucke|zeige)\b.{0,80}\b(above prompt|prompt|obige eingabeaufforderung|eingabeaufforderung)\b",
    },
    {
        "id": "secret_credential_request",
        "wall": "secret_exfiltration",
        "regex": r"\b(api key|token|password|credentials|secret)\b",
    },
    {
        "id": "tool_sql_db_access",
        "wall": "tool_or_action_abuse",
        "regex": r"\b(generate sql|sql\s*-?\s*code|access a database|database access|query database|datenbank|acceder a una base de datos)\b",
    },
    {
        "id": "tool_execute_command",
        "wall": "tool_or_action_abuse",
        "regex": r"\b(run command|execute command|execute|run shell|write file|network post)\b",
    },
    {
        "id": "evasion_bypass_safety",
        "wall": "policy_evasion",
        "regex": r"\b(bypass|jailbreak|sidestep|unlock mode)\b.{0,35}\b(safety|policy|guardrails|restrictions)\b",
    },
    {
        "id": "evasion_no_restrictions",
        "wall": "policy_evasion",
        "regex": r"\b(without restrictions|no restrictions|without limitations|do anything now|unrestricted)\b",
    },
    {
        "id": "soft_directive_i_want_you",
        "wall": "override_instructions",
        "regex": r"\b(i want you to|please|now you should|your task is|ich m[o\u00f6]chte dass sie|bitte|por favor|quiero que)\b",
    },
    {
        "id": "soft_directive_answer_with",
        "wall": "override_instructions",
        "regex": r"\b(answer\s+with|now,\s*answer me with|now\s+answer me with)\b",
    },
    {
        "id": "wrapper_text_answer_format",
        "wall": "override_instructions",
        "regex": r"\btext\s*:\s*.{0,220}\banswer\s*:",
    },
    {
        "id": "wrapper_role_prefix",
        "wall": "override_instructions",
        "regex": r"(?i)\b(system|developer|assistant|user)\s*:",
    },
)


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "you",
    "your",
    "are",
    "now",
    "only",
    "from",
    "into",
    "have",
    "will",
    "can",
    "please",
    "want",
    "act",
}


@dataclass(frozen=True)
class FNAnalysisRecord:
    sample_id: str
    split: str
    label: int
    is_attack: bool
    off: bool
    wall_any: bool
    predicted_attack: bool
    v_sum: float
    text: str
    fn_pattern_tags: List[str]
    suggested_wall: str


def _pattern_match(text: str) -> List[str]:
    t = str(text).strip()
    out: List[str] = []
    for spec in PATTERN_SPECS:
        if re.search(spec["regex"], t, flags=re.IGNORECASE | re.DOTALL):
            out.append(spec["id"])
    if not out:
        out.append("other_indirect")
    return sorted(set(out))


def _pattern_wall(pattern_id: str) -> str:
    for spec in PATTERN_SPECS:
        if spec["id"] == pattern_id:
            return spec["wall"]
    return "override_instructions"


def _guess_wall_from_text(text: str) -> str:
    t = text.lower()
    if any(k in t for k in ("token", "password", "credential", "secret", "prompt text", "embeddings", "eingabeaufforderung")):
        return "secret_exfiltration"
    if any(k in t for k in ("sql", "execute", "run command", "write file", "tool", "datenbank", "database")):
        return "tool_or_action_abuse"
    if any(k in t for k in ("bypass", "jailbreak", "no restrictions", "without restrictions", "guardrails")):
        return "policy_evasion"
    return "override_instructions"


def evaluate_deepset_records(
    *,
    projector: Pi0IntentAwareV2,
    omega_core: OmegaCoreV1,
    off_policy: OffPolicyV1,
    benchmark_root: str,
    split: str,
    mode: str,
    max_samples: int,
    seed: int,
    label_attack_value: int,
) -> List[FNAnalysisRecord]:
    bundle = build_deepset_samples(
        benchmark_root=benchmark_root,
        split=split,
        mode=mode,
        max_samples=max_samples,
        seed=seed,
        label_attack_value=label_attack_value,
    )
    records: List[FNAnalysisRecord] = []
    for sample in bundle.samples:
        item = ContentItem(
            doc_id=sample.sample_id,
            source_id=f"deepset:{split}",
            source_type="other",
            trust="untrusted",
            text=sample.text,
        )
        projection = projector.project(item)
        state = OmegaState(session_id=f"sess:{sample.sample_id}", m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        step_result = omega_core.step(state=state, items=[item], projections=[projection])
        decision = off_policy.select_actions(step_result=step_result, items=[item])
        pred_attack = bool(decision.off)
        fn_tags = _pattern_match(sample.text) if (sample.is_attack and not pred_attack) else []
        suggest = _guess_wall_from_text(sample.text)
        records.append(
            FNAnalysisRecord(
                sample_id=sample.sample_id,
                split=sample.split,
                label=int(sample.label),
                is_attack=bool(sample.is_attack),
                off=bool(decision.off),
                wall_any=bool(float(projection.v.sum()) > 0.0),
                predicted_attack=pred_attack,
                v_sum=float(projection.v.sum()),
                text=sample.text,
                fn_pattern_tags=fn_tags,
                suggested_wall=suggest,
            )
        )
    return records


def _extract_ngrams(texts: Iterable[str], *, n: int) -> Counter[str]:
    cnt: Counter[str] = Counter()
    for text in texts:
        toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3 and t not in STOPWORDS]
        if len(toks) < n:
            continue
        for i in range(0, len(toks) - n + 1):
            ng = " ".join(toks[i : i + n])
            cnt[ng] += 1
    return cnt


def summarize_fn_patterns(records: Sequence[FNAnalysisRecord], *, top_k: int = 30) -> Dict[str, Any]:
    attacks = [r for r in records if r.is_attack]
    fn_rows = [r for r in records if r.is_attack and not r.predicted_attack]
    pattern_counter: Counter[str] = Counter()
    examples: Dict[str, List[str]] = defaultdict(list)
    for row in fn_rows:
        tags = row.fn_pattern_tags if row.fn_pattern_tags else ["other_indirect"]
        for tag in tags:
            pattern_counter[tag] += 1
            if len(examples[tag]) < 3:
                examples[tag].append(" ".join(row.text.split())[:220])

    pattern_items: List[Dict[str, Any]] = []
    for tag, cnt in sorted(pattern_counter.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]:
        pattern_items.append(
            {
                "pattern_id": tag,
                "count": int(cnt),
                "suggested_wall": _pattern_wall(tag),
                "examples": examples.get(tag, []),
            }
        )

    fn_texts = [r.text for r in fn_rows]
    ng2 = _extract_ngrams(fn_texts, n=2)
    ng3 = _extract_ngrams(fn_texts, n=3)
    ngrams: List[Dict[str, Any]] = []
    for gram, cnt in sorted((ng3 + ng2).items(), key=lambda kv: (-kv[1], kv[0]))[: min(20, top_k)]:
        ngrams.append(
            {
                "pattern_id": f"ngram:{gram}",
                "count": int(cnt),
                "suggested_wall": _guess_wall_from_text(gram),
            }
        )

    return {
        "total_samples": len(records),
        "attack_total": len(attacks),
        "fn_total": len(fn_rows),
        "fn_rate": (float(len(fn_rows)) / float(len(attacks)) if attacks else 0.0),
        "top_patterns": pattern_items,
        "top_ngrams": ngrams,
    }


def write_fn_artifacts(*, out_dir: str, records: Sequence[FNAnalysisRecord], summary: Dict[str, Any]) -> Dict[str, str]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    samples_path = root / "fn_samples.jsonl"
    patterns_path = root / "fn_patterns.json"
    patterns_md_path = root / "fn_patterns.md"

    with samples_path.open("w", encoding="utf-8") as fh:
        for row in records:
            if row.is_attack and not row.predicted_attack:
                fh.write(
                    json.dumps(
                        {
                            "sample_id": row.sample_id,
                            "label": row.label,
                            "off": row.off,
                            "wall_any": row.wall_any,
                            "v_sum": row.v_sum,
                            "suggested_wall": row.suggested_wall,
                            "fn_pattern_tags": row.fn_pattern_tags,
                            "text": row.text,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )

    patterns_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    patterns_md_path.write_text(render_fn_patterns_markdown(summary), encoding="utf-8")
    return {
        "fn_samples_jsonl": samples_path.as_posix(),
        "fn_patterns_json": patterns_path.as_posix(),
        "fn_patterns_md": patterns_md_path.as_posix(),
    }


def render_fn_patterns_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Deepset FN Pattern Analysis",
        "",
        f"- total_samples: `{summary.get('total_samples', 0)}`",
        f"- attack_total: `{summary.get('attack_total', 0)}`",
        f"- fn_total: `{summary.get('fn_total', 0)}`",
        f"- fn_rate: `{float(summary.get('fn_rate', 0.0)):.6f}`",
        "",
        "## Top Patterns",
        "| pattern_id | count | suggested_wall |",
        "|---|---:|---|",
    ]
    for item in summary.get("top_patterns", []):
        lines.append(
            f"| `{item.get('pattern_id')}` | {int(item.get('count', 0))} | `{item.get('suggested_wall', '')}` |"
        )
    lines.extend(
        [
            "",
            "## Top N-grams",
            "| pattern_id | count | suggested_wall |",
            "|---|---:|---|",
        ]
    )
    for item in summary.get("top_ngrams", []):
        lines.append(
            f"| `{item.get('pattern_id')}` | {int(item.get('count', 0))} | `{item.get('suggested_wall', '')}` |"
        )
    lines.append("")
    return "\n".join(lines)
