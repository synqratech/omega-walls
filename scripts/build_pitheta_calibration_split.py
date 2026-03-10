from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.pitheta.dataset_builder import load_pitheta_jsonl

WALLS = [
    "override_instructions",
    "secret_exfiltration",
    "tool_or_action_abuse",
    "policy_evasion",
]


def _stable_priority(sample_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _is_ascii(text: str) -> bool:
    try:
        text.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def _stratum_key(row: Mapping[str, Any]) -> Tuple[int, str, str, str]:
    return (
        int(row.get("is_attack", 0)),
        str(row.get("source_type", "other")),
        str(row.get("chunk_bucket", "64")),
        "ascii" if _is_ascii(str(row.get("text", ""))) else "non_ascii",
    )


def _quota_by_group(
    groups: Mapping[Tuple[int, str, str, str], Sequence[Mapping[str, Any]]],
    target_n: int,
) -> Dict[Tuple[int, str, str, str], int]:
    total = sum(len(v) for v in groups.values())
    if total <= 0 or target_n <= 0:
        return {k: 0 for k in groups}
    target_n = min(int(target_n), total)
    raw: Dict[Tuple[int, str, str, str], float] = {}
    floor_q: Dict[Tuple[int, str, str, str], int] = {}
    for key, rows in groups.items():
        val = (len(rows) / total) * target_n
        raw[key] = val
        floor_q[key] = min(len(rows), int(math.floor(val)))
    allocated = sum(floor_q.values())
    need = target_n - allocated
    if need > 0:
        by_remainder = sorted(
            groups.keys(),
            key=lambda k: ((raw[k] - floor_q[k]), len(groups[k])),
            reverse=True,
        )
        for key in by_remainder:
            if need <= 0:
                break
            if floor_q[key] < len(groups[key]):
                floor_q[key] += 1
                need -= 1
    return floor_q


def _compute_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    input_counts: Mapping[str, int],
    source_splits: Sequence[str],
    seed: int,
    target_size: int,
    output_path: Path,
) -> Dict[str, Any]:
    source_type_counts: Dict[str, int] = {}
    source_trust_counts: Dict[str, int] = {}
    chunk_bucket_counts: Dict[str, int] = {}
    lang_counts: Dict[str, int] = {}
    ascii_counts: Dict[str, int] = {"ascii": 0, "non_ascii": 0}
    attack = 0
    benign = 0
    per_wall_activation: Dict[str, int] = {w: 0 for w in WALLS}
    per_wall_pressure: Dict[str, Dict[str, int]] = {w: {"0": 0, "1": 0, "2": 0, "3": 0} for w in WALLS}
    per_wall_polarity: Dict[str, Dict[str, int]] = {w: {"-1": 0, "0": 0, "1": 0} for w in WALLS}
    for row in rows:
        if int(row.get("is_attack", 0)) == 1:
            attack += 1
        else:
            benign += 1
        source_type = str(row.get("source_type", "other"))
        source_trust = str(row.get("source_trust", "untrusted"))
        chunk_bucket = str(row.get("chunk_bucket", "64"))
        lang = str(row.get("lang", "en"))
        source_type_counts[source_type] = source_type_counts.get(source_type, 0) + 1
        source_trust_counts[source_trust] = source_trust_counts.get(source_trust, 0) + 1
        chunk_bucket_counts[chunk_bucket] = chunk_bucket_counts.get(chunk_bucket, 0) + 1
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
        ascii_counts["ascii" if _is_ascii(str(row.get("text", ""))) else "non_ascii"] += 1
        labels = [int(x) for x in list(row.get("wall_labels", [0, 0, 0, 0]))]
        levels = [int(x) for x in list(row.get("pressure_level", [0, 0, 0, 0]))]
        pols = [int(x) for x in list(row.get("polarity", [0, 0, 0, 0]))]
        for i, wall in enumerate(WALLS):
            if labels[i] == 1:
                per_wall_activation[wall] += 1
            lvl = str(levels[i] if levels[i] in {0, 1, 2, 3} else 0)
            pol = str(pols[i] if pols[i] in {-1, 0, 1} else 0)
            per_wall_pressure[wall][lvl] = per_wall_pressure[wall].get(lvl, 0) + 1
            per_wall_polarity[wall][pol] = per_wall_polarity[wall].get(pol, 0) + 1

    sample_ids = sorted(str(r.get("sample_id", "")) for r in rows)
    sample_payload = "\n".join(sample_ids).encode("utf-8")
    rows_payload = "\n".join(
        json.dumps(dict(r), ensure_ascii=True, sort_keys=True) for r in sorted(rows, key=lambda x: str(x.get("sample_id", "")))
    ).encode("utf-8")
    warnings: List[str] = []
    if attack == 0:
        warnings.append("no_attack_rows_selected")
    if benign == 0:
        warnings.append("no_benign_rows_selected")
    for wall in WALLS:
        if per_wall_activation[wall] == 0:
            warnings.append(f"no_positive_activation:{wall}")

    return {
        "status": "ok",
        "schema_version": "pitheta_calibration_split_v1",
        "seed": int(seed),
        "source_splits": [str(x) for x in source_splits],
        "target_size": int(target_size),
        "selected_size": int(len(rows)),
        "input_counts_by_split": dict(input_counts),
        "output_path": output_path.as_posix(),
        "fingerprints": {
            "sample_ids_sha256": hashlib.sha256(sample_payload).hexdigest(),
            "records_sha256": hashlib.sha256(rows_payload).hexdigest(),
        },
        "distribution": {
            "attack": int(attack),
            "benign": int(benign),
            "source_type_counts": dict(sorted(source_type_counts.items())),
            "source_trust_counts": dict(sorted(source_trust_counts.items())),
            "chunk_bucket_counts": dict(sorted(chunk_bucket_counts.items())),
            "lang_counts": dict(sorted(lang_counts.items())),
            "ascii_non_ascii_counts": dict(sorted(ascii_counts.items())),
            "per_wall_activation_counts": per_wall_activation,
            "per_wall_pressure_level_counts": per_wall_pressure,
            "per_wall_polarity_counts": per_wall_polarity,
        },
        "warnings": warnings,
    }


def build_calibration_split(
    *,
    data_dir: str,
    output_path: str | None = None,
    report_path: str | None = None,
    source_splits: Sequence[str] = ("train", "dev"),
    target_size: int | None = None,
    target_ratio: float = 0.20,
    min_size: int = 200,
    seed: int = 41,
    strict: bool = False,
) -> Dict[str, Any]:
    root = Path(data_dir)
    split_rows: Dict[str, List[Dict[str, Any]]] = {}
    for split in source_splits:
        split_norm = str(split).strip().lower()
        if split_norm not in {"train", "dev", "holdout"}:
            raise ValueError("source_splits entries must be train|dev|holdout")
        path = root / f"{split_norm}.jsonl"
        if not path.exists():
            if strict:
                raise ValueError(f"missing split file: {path.as_posix()}")
            continue
        split_rows[split_norm] = load_pitheta_jsonl(path.as_posix())

    if not split_rows:
        raise ValueError("no input rows loaded from requested source_splits")

    all_rows: List[Dict[str, Any]] = []
    input_counts: Dict[str, int] = {}
    seen_ids: set[str] = set()
    for split in source_splits:
        split_norm = str(split).strip().lower()
        rows = split_rows.get(split_norm, [])
        input_counts[split_norm] = len(rows)
        for row in rows:
            sid = str(row.get("sample_id", ""))
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
            all_rows.append(dict(row))

    if not all_rows:
        raise ValueError("input rows are empty after deduplication")

    if target_size is not None:
        computed_target = int(target_size)
    else:
        computed_target = max(int(min_size), int(round(len(all_rows) * float(target_ratio))))
    computed_target = min(computed_target, len(all_rows))

    grouped: Dict[Tuple[int, str, str, str], List[Dict[str, Any]]] = {}
    for row in all_rows:
        key = _stratum_key(row)
        grouped.setdefault(key, []).append(row)
    for key, rows in grouped.items():
        grouped[key] = sorted(rows, key=lambda r: _stable_priority(str(r.get("sample_id", "")), seed))

    quotas = _quota_by_group(grouped, computed_target)
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    for key in sorted(grouped.keys()):
        rows = grouped[key]
        take_n = min(len(rows), int(quotas.get(key, 0)))
        for row in rows[:take_n]:
            sid = str(row.get("sample_id", ""))
            if sid in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(sid)

    if len(selected) < computed_target:
        remaining = sorted(
            (r for r in all_rows if str(r.get("sample_id", "")) not in selected_ids),
            key=lambda r: _stable_priority(str(r.get("sample_id", "")), seed),
        )
        need = computed_target - len(selected)
        selected.extend(remaining[:need])

    selected = sorted(selected, key=lambda r: _stable_priority(str(r.get("sample_id", "")), seed))
    output = Path(output_path) if output_path else (root / "calibration.jsonl")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for row in selected:
            fh.write(json.dumps(dict(row), ensure_ascii=True) + "\n")

    report = _compute_report(
        selected,
        input_counts=input_counts,
        source_splits=source_splits,
        seed=seed,
        target_size=computed_target,
        output_path=output,
    )
    report_out = Path(report_path) if report_path else output.with_name("calibration_report.json")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    report["report_path"] = report_out.as_posix()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic calibration.jsonl with stratified sampling and coverage report.")
    parser.add_argument("--data-dir", required=True, help="Directory with train.jsonl/dev.jsonl/holdout.jsonl")
    parser.add_argument("--output-path", default=None, help="Output calibration.jsonl path (default: <data-dir>/calibration.jsonl)")
    parser.add_argument("--report-path", default=None, help="Output report path (default: alongside output)")
    parser.add_argument("--source-splits", default="train,dev", help="Comma-separated source splits: train,dev,holdout")
    parser.add_argument("--target-size", type=int, default=None, help="Exact target size for calibration split")
    parser.add_argument("--target-ratio", type=float, default=0.20, help="Target ratio of source pool if --target-size is omitted")
    parser.add_argument("--min-size", type=int, default=200, help="Minimum selected rows")
    parser.add_argument("--seed", type=int, default=41, help="Deterministic seed")
    parser.add_argument("--strict", action="store_true", help="Fail if requested split files are missing")
    args = parser.parse_args()

    source_splits = [x.strip().lower() for x in str(args.source_splits).split(",") if x.strip()]
    if not source_splits:
        raise ValueError("source_splits must be non-empty")
    report = build_calibration_split(
        data_dir=str(args.data_dir),
        output_path=args.output_path,
        report_path=args.report_path,
        source_splits=source_splits,
        target_size=args.target_size,
        target_ratio=float(args.target_ratio),
        min_size=int(args.min_size),
        seed=int(args.seed),
        strict=bool(args.strict),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
