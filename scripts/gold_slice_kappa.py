from __future__ import annotations

import argparse
import json
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.pitheta.gold_slice import compute_gold_slice_agreement, load_gold_slice_jsonl


def _jsonl_write(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=True) + "\n")


def _rows_digest(rows: Sequence[Dict[str, Any]]) -> str:
    normalized = []
    for row in rows:
        normalized.append(
            {
                "sample_id": str(row.get("sample_id", "")),
                "wall_labels": [int(x) for x in list(row.get("wall_labels", [0, 0, 0, 0]))],
                "pressure_level": [int(x) for x in list(row.get("pressure_level", [0, 0, 0, 0]))],
                "polarity": [int(x) for x in list(row.get("polarity", [0, 0, 0, 0]))],
            }
        )
    payload = "\n".join(
        json.dumps(row, ensure_ascii=True, sort_keys=True)
        for row in sorted(normalized, key=lambda x: str(x["sample_id"]))
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _to_md(report: Mapping[str, Any]) -> str:
    agreement = report.get("agreement", {}) or {}
    ord_kappa = agreement.get("ordinal_quadratic_kappa_per_wall", {}) or {}
    pol_kappa = agreement.get("polarity_quadratic_kappa_per_wall", {}) or {}
    lines = [
        "# Gold Slice Agreement Report",
        "",
        f"- matched_count: `{agreement.get('matched_count', 0)}`",
        f"- exact_match_rate: `{agreement.get('exact_match_rate', 0.0):.4f}`",
        f"- only_in_annotator_a: `{agreement.get('only_in_annotator_a', 0)}`",
        f"- only_in_annotator_b: `{agreement.get('only_in_annotator_b', 0)}`",
        f"- identical_annotation_hash: `{bool((report.get('independence', {}) or {}).get('identical_annotations', False))}`",
        "",
        "## Ordinal Kappa (quadratic)",
    ]
    for wall, value in ord_kappa.items():
        lines.append(f"- `{wall}`: `{float(value):.4f}`")
    lines.append("")
    lines.append("## Polarity Kappa (quadratic)")
    for wall, value in pol_kappa.items():
        lines.append(f"- `{wall}`: `{float(value):.4f}`")
    lines.append("")
    lines.append(f"- adjudication_rows: `{report.get('adjudication_rows', 0)}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute quadratic kappa and adjudication report for gold-slice annotations.")
    parser.add_argument("--annotator-a", required=True, help="Path to annotator A JSONL")
    parser.add_argument("--annotator-b", required=True, help="Path to annotator B JSONL")
    parser.add_argument("--output-dir", default=None, help="Artifacts output directory")
    parser.add_argument("--top-k-adjudication", type=int, default=200)
    parser.add_argument("--strict-thresholds", action="store_true")
    parser.add_argument("--ordinal-threshold", type=float, default=0.70)
    parser.add_argument("--polarity-threshold", type=float, default=0.65)
    parser.add_argument("--require-independent", action="store_true")
    args = parser.parse_args()

    rows_a = load_gold_slice_jsonl(str(args.annotator_a))
    rows_b = load_gold_slice_jsonl(str(args.annotator_b))
    agreement, adjudication = compute_gold_slice_agreement(rows_a, rows_b)
    digest_a = _rows_digest(rows_a)
    digest_b = _rows_digest(rows_b)
    identical_annotations = bool(digest_a == digest_b and len(rows_a) > 0 and len(rows_b) > 0)
    if int(args.top_k_adjudication) > 0:
        adjudication = adjudication[: int(args.top_k_adjudication)]

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir or (ROOT / "artifacts" / "gold_slice_agreement" / f"run_{run_id}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "status": "ok",
        "run_id": run_id,
        "annotator_a_path": str(Path(args.annotator_a).as_posix()),
        "annotator_b_path": str(Path(args.annotator_b).as_posix()),
        "agreement": agreement,
        "independence": {
            "require_independent": bool(args.require_independent),
            "annotator_a_digest": digest_a,
            "annotator_b_digest": digest_b,
            "identical_annotations": identical_annotations,
        },
        "thresholds": {
            "ordinal": float(args.ordinal_threshold),
            "polarity": float(args.polarity_threshold),
        },
        "adjudication_rows": int(len(adjudication)),
        "artifacts": {
            "agreement_report_json": (out_dir / "agreement_report.json").as_posix(),
            "adjudication_report_jsonl": (out_dir / "adjudication_report.jsonl").as_posix(),
            "agreement_report_md": (out_dir / "agreement_report.md").as_posix(),
        },
    }
    (out_dir / "agreement_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    _jsonl_write(out_dir / "adjudication_report.jsonl", adjudication)
    (out_dir / "agreement_report.md").write_text(_to_md(report), encoding="utf-8")
    failures = []
    ord_map = agreement.get("ordinal_quadratic_kappa_per_wall", {}) if isinstance(agreement, dict) else {}
    pol_map = agreement.get("polarity_quadratic_kappa_per_wall", {}) if isinstance(agreement, dict) else {}
    for wall, value in ord_map.items():
        if float(value) < float(args.ordinal_threshold):
            failures.append(f"ordinal:{wall}:{value:.4f}")
    for wall, value in pol_map.items():
        if float(value) < float(args.polarity_threshold):
            failures.append(f"polarity:{wall}:{value:.4f}")
    if bool(args.require_independent) and identical_annotations:
        failures.append("independence:annotator_a_and_b_identical")
    report["status"] = "PASS" if not failures else "FAIL"
    report["failures"] = failures
    (out_dir / "agreement_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.strict_thresholds and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
