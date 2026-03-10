"""Weak-label audit utilities for PiTheta train datasets."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2


def _is_ascii(text: str) -> bool:
    return all(ord(ch) < 128 for ch in str(text))


def _wall_any(walls: Sequence[int]) -> int:
    return 1 if any(int(x) > 0 for x in walls) else 0


def _pick_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_size: int,
    seed: int,
    include_qualities: Sequence[str],
) -> List[Dict[str, Any]]:
    include_set = {str(x).strip().lower() for x in include_qualities if str(x).strip()}
    candidates = []
    for row in rows:
        quality = str(row.get("label_quality", "")).strip().lower()
        if include_set and quality not in include_set:
            continue
        candidates.append(dict(row))

    rng = random.Random(seed)
    if len(candidates) <= int(sample_size):
        out = list(candidates)
        rng.shuffle(out)
        return out
    idx = sorted(rng.sample(range(len(candidates)), k=int(sample_size)))
    out = [candidates[i] for i in idx]
    rng.shuffle(out)
    return out


def _safe_rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def evaluate_weak_label_audit(
    *,
    projector: Pi0IntentAwareV2,
    rows: Sequence[Mapping[str, Any]],
    sample_size: int,
    seed: int,
    include_qualities: Sequence[str],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    sampled = _pick_rows(
        rows,
        sample_size=int(sample_size),
        seed=int(seed),
        include_qualities=include_qualities,
    )
    walls = list(WALLS_V1)

    wall_match_counts = [0, 0, 0, 0]
    polarity_match_counts = [0, 0, 0, 0]
    exact_wall_rows = 0
    exact_polarity_rows = 0
    mismatch_rows = 0
    attack_any_old = 0
    attack_any_new = 0
    attack_any_match = 0
    ascii_count = 0
    non_ascii_count = 0
    quality_counts: Dict[str, int] = {}
    quality_mismatch_counts: Dict[str, int] = {}

    review_rows: List[Dict[str, Any]] = []
    for row in sampled:
        text = str(row.get("text", "")).strip()
        doc_id = str(row.get("sample_id", "unknown"))
        source = str(row.get("source", "unknown"))
        quality = str(row.get("label_quality", "unknown"))
        old_walls = [int(x) for x in list(row.get("wall_labels", [0, 0, 0, 0]))]
        old_pol = [int(x) for x in list(row.get("polarity", [0, 0, 0, 0]))]
        if _is_ascii(text):
            ascii_count += 1
        else:
            non_ascii_count += 1
        quality_counts[quality] = quality_counts.get(quality, 0) + 1

        proj = projector.project(
            ContentItem(
                doc_id=doc_id,
                source_id=source,
                source_type="other",
                trust="untrusted",
                text=text,
            )
        )
        new_walls = [1 if float(x) > 0.0 else 0 for x in proj.v.tolist()]
        new_pol = [int(x) for x in proj.evidence.polarity]

        wall_match_flags = [int(old_walls[i] == new_walls[i]) for i in range(4)]
        pol_match_flags = [int(old_pol[i] == new_pol[i]) for i in range(4)]
        for i in range(4):
            wall_match_counts[i] += wall_match_flags[i]
            polarity_match_counts[i] += pol_match_flags[i]

        wall_exact = all(bool(x) for x in wall_match_flags)
        pol_exact = all(bool(x) for x in pol_match_flags)
        if wall_exact:
            exact_wall_rows += 1
        if pol_exact:
            exact_polarity_rows += 1

        old_any = _wall_any(old_walls)
        new_any = _wall_any(new_walls)
        attack_any_old += old_any
        attack_any_new += new_any
        if old_any == new_any:
            attack_any_match += 1

        mismatch_count = sum(1 for i in range(4) if old_walls[i] != new_walls[i]) + sum(
            1 for i in range(4) if old_pol[i] != new_pol[i]
        )
        mismatch_any = mismatch_count > 0
        if mismatch_any:
            mismatch_rows += 1
            quality_mismatch_counts[quality] = quality_mismatch_counts.get(quality, 0) + 1

        high_risk = (
            mismatch_count >= 3
            or (old_any == 1 and new_any == 0)
            or (old_any == 0 and new_any == 1)
            or (not _is_ascii(text))
        )

        review_rows.append(
            {
                "sample_id": doc_id,
                "source": source,
                "label_quality": quality,
                "is_attack": int(row.get("is_attack", 0)),
                "is_ascii": bool(_is_ascii(text)),
                "text": text,
                "old_wall_labels": old_walls,
                "new_wall_labels": new_walls,
                "old_polarity": old_pol,
                "new_polarity": new_pol,
                "wall_match_flags": wall_match_flags,
                "polarity_match_flags": pol_match_flags,
                "mismatch_count": int(mismatch_count),
                "mismatch_any": bool(mismatch_any),
                "attack_any_old": int(old_any),
                "attack_any_new": int(new_any),
                "priority_review": bool(high_risk),
            }
        )

    n = len(sampled)
    summary = {
        "sampled_count": n,
        "sample_size_requested": int(sample_size),
        "seed": int(seed),
        "ascii_count": int(ascii_count),
        "non_ascii_count": int(non_ascii_count),
        "label_quality_counts": dict(sorted(quality_counts.items())),
        "label_quality_mismatch_counts": dict(sorted(quality_mismatch_counts.items())),
        "agreement": {
            "exact_wall_rows_rate": _safe_rate(exact_wall_rows, n),
            "exact_polarity_rows_rate": _safe_rate(exact_polarity_rows, n),
            "any_mismatch_rows_rate": _safe_rate(mismatch_rows, n),
            "attack_any_agreement_rate": _safe_rate(attack_any_match, n),
            "attack_any_old_rate": _safe_rate(attack_any_old, n),
            "attack_any_new_rate": _safe_rate(attack_any_new, n),
            "wall_label_per_wall_rate": {
                walls[i]: _safe_rate(wall_match_counts[i], n) for i in range(4)
            },
            "polarity_per_wall_rate": {
                walls[i]: _safe_rate(polarity_match_counts[i], n) for i in range(4)
            },
        },
    }
    return summary, review_rows


def write_weak_label_audit_artifacts(
    *,
    out_dir: str,
    report: Mapping[str, Any],
    review_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    report_path = out / "audit_report.json"
    review_path = out / "review_samples.jsonl"
    priority_path = out / "priority_review_samples.jsonl"
    summary_md_path = out / "audit_summary.md"

    report_path.write_text(json.dumps(dict(report), ensure_ascii=True, indent=2), encoding="utf-8")
    _jsonl_write(review_path, review_rows)
    _jsonl_write(priority_path, [r for r in review_rows if bool(r.get("priority_review", False))])

    lines = [
        "# PiTheta Weak Label Audit",
        "",
        f"- sampled_count: `{report.get('sampled_count')}`",
        f"- ascii_count: `{report.get('ascii_count')}`",
        f"- non_ascii_count: `{report.get('non_ascii_count')}`",
        f"- exact_wall_rows_rate: `{(report.get('agreement', {}) or {}).get('exact_wall_rows_rate')}`",
        f"- exact_polarity_rows_rate: `{(report.get('agreement', {}) or {}).get('exact_polarity_rows_rate')}`",
        f"- any_mismatch_rows_rate: `{(report.get('agreement', {}) or {}).get('any_mismatch_rows_rate')}`",
    ]
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "audit_report": report_path.as_posix(),
        "review_samples": review_path.as_posix(),
        "priority_review_samples": priority_path.as_posix(),
        "audit_summary_md": summary_md_path.as_posix(),
    }

