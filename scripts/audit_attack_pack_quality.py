from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.build_attack_layer_api_pack_v1 import (
        ATTACK_REFUSAL_PATTERNS,
        BENIGN_SESSION_ACTION_CUE_PATTERNS,
        FAMILY_INTENT_PROGRESSION_PATTERNS,
        REAL_OBFUSCATION_PATTERNS,
        TARGET_BENIGN_NONIMPERATIVE_FAMILIES,
    )
except Exception:
    from build_attack_layer_api_pack_v1 import (
        ATTACK_REFUSAL_PATTERNS,
        BENIGN_SESSION_ACTION_CUE_PATTERNS,
        FAMILY_INTENT_PROGRESSION_PATTERNS,
        REAL_OBFUSCATION_PATTERNS,
        TARGET_BENIGN_NONIMPERATIVE_FAMILIES,
    )

OBF_FAMILIES = {"token_smuggling", "semantic_obfuscation"}
TARGET_AUDIT_FAMILIES = set(FAMILY_INTENT_PROGRESSION_PATTERNS.keys()) | set(TARGET_BENIGN_NONIMPERATIVE_FAMILIES)


@dataclass(frozen=True)
class Row:
    case_id: str
    layer: str
    family: str
    mode: str
    turn_id: int
    turn_label: str
    expected_off: bool
    input_text: str
    tool_output_text: str


def _norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def _normalize_apostrophes(value: Any) -> str:
    text = str(value or "")
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u0060", "'")
        .replace("\u00b4", "'")
        .replace("\u2032", "'")
    )


def _read_jsonl(path: Path) -> List[Row]:
    rows: List[Row] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = str(line).strip()
        if not ln:
            continue
        obj = json.loads(ln)
        if not isinstance(obj, dict):
            continue
        rows.append(
            Row(
                case_id=str(obj.get("case_id", "")).strip(),
                layer=str(obj.get("layer", "")).strip(),
                family=str(obj.get("family", "")).strip(),
                mode=str(obj.get("mode", "")).strip(),
                turn_id=int(obj.get("turn_id", 0)),
                turn_label=str(obj.get("turn_label", "")).strip(),
                expected_off=bool(obj.get("expected_off", False)),
                input_text=_norm(obj.get("input_text", "")),
                tool_output_text=_norm(obj.get("tool_output_text", "")),
            )
        )
    rows.sort(key=lambda r: (r.case_id, r.turn_id))
    return rows


def _top_ngrams(texts: Sequence[str], *, n: int = 4, top_k: int = 12) -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    total = max(1, len(texts))
    for text in texts:
        tokens = re.findall(r"[A-Za-z0-9_]+", str(text).lower())
        if len(tokens) < n:
            continue
        seen_local: set[str] = set()
        for i in range(0, len(tokens) - n + 1):
            ng = " ".join(tokens[i : i + n])
            if ng in seen_local:
                continue
            counts[ng] = int(counts.get(ng, 0)) + 1
            seen_local.add(ng)
    ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:top_k]
    return [{"ngram": ng, "count": cnt, "share": float(cnt) / float(total)} for ng, cnt in ranked]


def _has_obfuscation_signal(text: str) -> bool:
    return any(bool(p.search(str(text or ""))) for p in REAL_OBFUSCATION_PATTERNS)


def _contains_any(text: str, patterns: Sequence[re.Pattern[str]]) -> bool:
    return any(bool(p.search(str(text or ""))) for p in patterns)


def audit_manifest(rows: Sequence[Row]) -> Dict[str, Any]:
    by_case: Dict[str, List[Row]] = defaultdict(list)
    for row in rows:
        by_case[row.case_id].append(row)

    attack_rows = [r for r in rows if r.expected_off]
    benign_rows = [r for r in rows if not r.expected_off]
    attack_texts = [f"{r.input_text} {r.tool_output_text}".strip() for r in attack_rows]
    benign_texts = [f"{r.input_text} {r.tool_output_text}".strip() for r in benign_rows]

    refusal_rows = 0
    for text in attack_texts:
        norm = _normalize_apostrophes(text)
        if _contains_any(norm, ATTACK_REFUSAL_PATTERNS):
            refusal_rows += 1

    stage_issues: List[Dict[str, Any]] = []
    benign_nonimperative_violations: List[Dict[str, Any]] = []
    for case_id, case_rows in by_case.items():
        if not case_rows or case_rows[0].mode != "session":
            continue
        ordered = sorted(case_rows, key=lambda r: r.turn_id)
        labels = [r.turn_label for r in ordered]
        core_label = "exploit" if bool(ordered[0].expected_off) else "benign_response"
        expected_labels = ["benign_seed", "bridge", core_label, "followup"]
        if labels != expected_labels:
            stage_issues.append({"case_id": case_id, "issue": "label_order", "labels": labels})
            continue
        exploit = next((r.input_text for r in ordered if r.turn_label == core_label), "")
        followup = next((r.input_text for r in ordered if r.turn_label == "followup"), "")
        if exploit and followup and exploit == followup:
            stage_issues.append({"case_id": case_id, "issue": "exploit_followup_identical"})

        if (not bool(ordered[0].expected_off)) and (ordered[0].family in TARGET_BENIGN_NONIMPERATIVE_FAMILIES):
            check_rows = [r for r in ordered if r.turn_label in {"bridge", "followup"}]
            stage_text = _normalize_apostrophes(" ".join(f"{r.input_text} {r.tool_output_text}".strip() for r in check_rows))
            if _contains_any(stage_text, BENIGN_SESSION_ACTION_CUE_PATTERNS):
                benign_nonimperative_violations.append(
                    {
                        "case_id": case_id,
                        "family": ordered[0].family,
                        "issue": "benign_session_nonimperative",
                    }
                )

    obf_rows = [r for r in attack_rows if r.layer == "obfuscation" and r.family in OBF_FAMILIES]
    obf_with_signal = 0
    for r in obf_rows:
        blob = f"{r.input_text} {r.tool_output_text}"
        if _has_obfuscation_signal(blob):
            obf_with_signal += 1

    family_slices: Dict[str, Any] = {}
    for family in sorted(TARGET_AUDIT_FAMILIES):
        fam_rows = [r for r in rows if r.family == family]
        fam_attack_rows = [r for r in fam_rows if r.expected_off]
        fam_benign_rows = [r for r in fam_rows if not r.expected_off]
        fam_attack_texts = [f"{r.input_text} {r.tool_output_text}".strip() for r in fam_attack_rows]
        fam_refusal_rows = 0
        for text in fam_attack_texts:
            if _contains_any(_normalize_apostrophes(text), ATTACK_REFUSAL_PATTERNS):
                fam_refusal_rows += 1
        family_slices[family] = {
            "rows_total": int(len(fam_rows)),
            "attack_rows": int(len(fam_attack_rows)),
            "benign_rows": int(len(fam_benign_rows)),
            "attack_refusal_rows": int(fam_refusal_rows),
            "attack_refusal_rate": float(fam_refusal_rows) / float(len(fam_attack_rows) or 1),
            "top_attack_ngrams": _top_ngrams(fam_attack_texts, n=4, top_k=6),
        }

    return {
        "summary": {
            "rows_total": len(rows),
            "cases_total": len(by_case),
            "attack_rows": len(attack_rows),
            "benign_rows": len(benign_rows),
        },
        "template_pressure": {
            "top_attack_ngrams": _top_ngrams(attack_texts, n=4, top_k=12),
            "top_benign_ngrams": _top_ngrams(benign_texts, n=4, top_k=12),
        },
        "refusal_contamination": {
            "attack_rows_with_refusal": int(refusal_rows),
            "attack_refusal_rate": float(refusal_rows) / float(len(attack_rows) or 1),
        },
        "stage_progression": {
            "session_cases_total": sum(1 for rows_ in by_case.values() if rows_ and rows_[0].mode == "session"),
            "cases_with_issues": len(stage_issues),
            "issues": stage_issues[:50],
        },
        "benign_session_nonimperative": {
            "target_families": sorted(TARGET_BENIGN_NONIMPERATIVE_FAMILIES),
            "violations": int(len(benign_nonimperative_violations)),
            "cases": benign_nonimperative_violations[:50],
        },
        "obfuscation_realism": {
            "rows_checked": len(obf_rows),
            "rows_with_signal": int(obf_with_signal),
            "signal_rate": float(obf_with_signal) / float(len(obf_rows) or 1),
        },
        "family_target_slices": family_slices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated attack-pack quality (read-only).")
    parser.add_argument("--pack-root", default="tests/data/attack_layers/v1_api_main_fix1")
    parser.add_argument("--manifest", default=None, help="Optional explicit manifest path.")
    parser.add_argument("--out", default=None, help="Optional JSON output file.")
    args = parser.parse_args()

    manifest = Path(args.manifest).resolve() if args.manifest else (ROOT / str(args.pack_root) / "manifest_all.jsonl").resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"manifest not found: {manifest}")

    rows = _read_jsonl(manifest)
    report = {
        "manifest": str(manifest),
        "status": "ok",
        "audit": audit_manifest(rows),
    }
    if args.out:
        out_path = (ROOT / str(args.out)).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
