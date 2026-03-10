from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem
from omega.projector.factory import build_projector
from omega.rag.attachment_ingestion import AttachmentExtractResult, extract_attachment


@dataclass(frozen=True)
class EvalRow:
    sample_id: str
    path: str
    label: int
    format: str
    note: str
    expected_scan_like: Optional[bool] = None


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _load_manifest(path: Path) -> List[EvalRow]:
    rows: List[EvalRow] = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        line = ln.strip()
        if not line:
            continue
        raw = json.loads(line)
        rows.append(
            EvalRow(
                sample_id=str(raw.get("id", f"sample-{len(rows):04d}")),
                path=str(raw["path"]),
                label=int(raw["label"]),
                format=str(raw.get("format", "")).strip().lower(),
                note=str(raw.get("note", "")),
                expected_scan_like=(
                    bool(raw["expected_scan_like"])
                    if "expected_scan_like" in raw and raw.get("expected_scan_like") is not None
                    else None
                ),
            )
        )
    return rows


def _run_projector_on_chunks(projector: Any, chunks: Iterable[str], sample_id: str, source_id: str, source_type: str) -> bool:
    for idx, chunk in enumerate(chunks):
        item = ContentItem(
            doc_id=f"{sample_id}:c{idx:03d}",
            source_id=source_id,
            source_type=source_type,
            trust="untrusted",
            text=str(chunk),
        )
        out = projector.project(item)
        if float(np.sum(out.v)) > 0.0:
            return True
    return False


def _metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    attack_total = tp + fn
    benign_total = tn + fp
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "attack_off_rate": _safe_div(tp, attack_total),
        "benign_off_rate": _safe_div(fp, benign_total),
        "attack_total": int(attack_total),
        "benign_total": int(benign_total),
    }


def _deferred_policy_reasons(*, extracted: AttachmentExtractResult, fmt_key: str) -> List[str]:
    reasons: List[str] = []
    warnings = set(str(x).strip().lower() for x in list(extracted.warnings or []))
    extracted_fmt = str(getattr(extracted, "format", "")).strip().lower()
    if ("zip_deferred_runtime" in warnings) or (fmt_key == "zip") or (extracted_fmt == "zip"):
        reasons.append("zip_deferred_runtime")
    if bool(getattr(extracted, "scan_like", False)):
        reasons.append("scan_like")
    if bool(getattr(extracted, "text_empty", False)):
        reasons.append("text_empty")
    return reasons


def evaluate_attachment_manifest(
    *,
    manifest_rows: List[EvalRow],
    manifest_dir: Path,
    projector: Any,
    attachment_cfg: Mapping[str, Any] | None,
    use_recommended_verdict: bool = True,
) -> Dict[str, Any]:
    by_format_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    by_format_total: Dict[str, int] = defaultdict(int)
    by_format_parse_ok: Dict[str, int] = defaultdict(int)
    by_format_text_empty: Dict[str, int] = defaultdict(int)
    by_format_scan_like: Dict[str, int] = defaultdict(int)
    by_format_scan_like_expected_total: Dict[str, int] = defaultdict(int)
    by_format_scan_like_expected_hit: Dict[str, int] = defaultdict(int)
    by_format_quarantine_recommended: Dict[str, int] = defaultdict(int)
    core_counts: Dict[str, int] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    deferred_counts: Dict[str, int] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    deferred_reason_counts: Dict[str, int] = defaultdict(int)
    core_total = 0
    deferred_total = 0
    errors: List[Dict[str, Any]] = []

    for row in manifest_rows:
        path_abs = (manifest_dir / row.path).resolve()
        fmt_key = row.format or path_abs.suffix.lower().lstrip(".") or "unknown"
        by_format_total[fmt_key] += 1
        predicted_off = False
        is_deferred_policy = False
        deferred_reasons: List[str] = []

        try:
            extracted: AttachmentExtractResult = extract_attachment(path=path_abs, cfg=attachment_cfg)
            by_format_parse_ok[fmt_key] += 1
            if extracted.text_empty:
                by_format_text_empty[fmt_key] += 1
            if extracted.scan_like:
                by_format_scan_like[fmt_key] += 1
            if extracted.recommended_verdict != "allow":
                by_format_quarantine_recommended[fmt_key] += 1
            deferred_reasons = _deferred_policy_reasons(extracted=extracted, fmt_key=fmt_key)
            is_deferred_policy = bool(deferred_reasons)
            if row.expected_scan_like is True:
                by_format_scan_like_expected_total[fmt_key] += 1
                if extracted.scan_like:
                    by_format_scan_like_expected_hit[fmt_key] += 1
            chunk_texts = [c.text for c in extracted.chunks]
            projector_hit = _run_projector_on_chunks(
                projector=projector,
                chunks=chunk_texts,
                sample_id=row.sample_id,
                source_id=f"attachment:{fmt_key}",
                source_type=fmt_key if fmt_key in {"pdf", "docx", "html", "zip"} else "other",
            )
            verdict_hit = bool(use_recommended_verdict and extracted.recommended_verdict != "allow")
            predicted_off = bool(projector_hit or verdict_hit)
        except Exception as exc:
            errors.append({"id": row.sample_id, "format": fmt_key, "path": str(path_abs), "error": str(exc)})
            predicted_off = False
            if fmt_key == "zip":
                deferred_reasons = ["zip_deferred_runtime"]
                is_deferred_policy = True

        if row.label == 1 and predicted_off:
            by_format_counts[fmt_key]["tp"] += 1
            if is_deferred_policy:
                deferred_counts["tp"] += 1
            else:
                core_counts["tp"] += 1
        elif row.label == 1 and not predicted_off:
            by_format_counts[fmt_key]["fn"] += 1
            if is_deferred_policy:
                deferred_counts["fn"] += 1
            else:
                core_counts["fn"] += 1
        elif row.label != 1 and predicted_off:
            by_format_counts[fmt_key]["fp"] += 1
            if is_deferred_policy:
                deferred_counts["fp"] += 1
            else:
                core_counts["fp"] += 1
        else:
            by_format_counts[fmt_key]["tn"] += 1
            if is_deferred_policy:
                deferred_counts["tn"] += 1
            else:
                core_counts["tn"] += 1
        if is_deferred_policy:
            deferred_total += 1
            for reason in deferred_reasons:
                deferred_reason_counts[str(reason)] += 1
        else:
            core_total += 1

    per_format: Dict[str, Any] = {}
    total_tp = total_fp = total_tn = total_fn = 0
    for fmt in sorted(by_format_total.keys()):
        counts = by_format_counts[fmt]
        total_tp += counts["tp"]
        total_fp += counts["fp"]
        total_tn += counts["tn"]
        total_fn += counts["fn"]
        total = by_format_total[fmt]
        parse_ok = by_format_parse_ok[fmt]
        per_format[fmt] = {
            **_metrics_from_counts(counts["tp"], counts["fp"], counts["tn"], counts["fn"]),
            "total": int(total),
            "parse_success_rate": _safe_div(parse_ok, total),
            "text_empty_rate": _safe_div(by_format_text_empty[fmt], total),
            "scan_like_rate": _safe_div(by_format_scan_like[fmt], total),
            "quarantine_recommended_rate": _safe_div(by_format_quarantine_recommended[fmt], total),
            "scan_like_expected_total": int(by_format_scan_like_expected_total[fmt]),
            "scan_like_recall": _safe_div(
                by_format_scan_like_expected_hit[fmt],
                by_format_scan_like_expected_total[fmt],
            ),
        }

    total_scan_like_expected = int(sum(by_format_scan_like_expected_total.values()))
    total_scan_like_hit = int(sum(by_format_scan_like_expected_hit.values()))
    return {
        "summary": {
            **_metrics_from_counts(total_tp, total_fp, total_tn, total_fn),
            "total": int(len(manifest_rows)),
            "parse_success_rate": _safe_div(sum(by_format_parse_ok.values()), len(manifest_rows)),
            "text_empty_rate": _safe_div(sum(by_format_text_empty.values()), len(manifest_rows)),
            "scan_like_rate": _safe_div(sum(by_format_scan_like.values()), len(manifest_rows)),
            "quarantine_recommended_rate": _safe_div(sum(by_format_quarantine_recommended.values()), len(manifest_rows)),
            "scan_like_expected_total": total_scan_like_expected,
            "scan_like_recall": _safe_div(total_scan_like_hit, total_scan_like_expected),
        },
        "summary_core": {
            **_metrics_from_counts(core_counts["tp"], core_counts["fp"], core_counts["tn"], core_counts["fn"]),
            "total": int(core_total),
        },
        "summary_deferred_policy": {
            **_metrics_from_counts(
                deferred_counts["tp"],
                deferred_counts["fp"],
                deferred_counts["tn"],
                deferred_counts["fn"],
            ),
            "total": int(deferred_total),
        },
        "deferred_reasons_breakdown": {
            "zip_deferred_runtime": int(deferred_reason_counts.get("zip_deferred_runtime", 0)),
            "scan_like": int(deferred_reason_counts.get("scan_like", 0)),
            "text_empty": int(deferred_reason_counts.get("text_empty", 0)),
        },
        "per_format": per_format,
        "errors": errors,
    }


def evaluate_gate(
    *,
    summary: Mapping[str, Any],
    summary_label: str,
    per_format: Mapping[str, Any],
    required_formats: Sequence[str],
    attack_off_rate_ge: float,
    benign_off_rate_le: float,
    precision_ge: float,
    recall_ge: float,
    per_format_parse_success_ge: float,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def _check(name: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual, "expected": expected})

    attack_off_rate = float(summary.get("attack_off_rate", 0.0))
    benign_off_rate = float(summary.get("benign_off_rate", 1.0))
    precision = float(summary.get("precision", 0.0))
    recall = float(summary.get("recall", 0.0))
    _check(f"{summary_label}.attack_off_rate_ge", attack_off_rate >= attack_off_rate_ge, attack_off_rate, attack_off_rate_ge)
    _check(f"{summary_label}.benign_off_rate_le", benign_off_rate <= benign_off_rate_le, benign_off_rate, benign_off_rate_le)
    _check(f"{summary_label}.precision_ge", precision >= precision_ge, precision, precision_ge)
    _check(f"{summary_label}.recall_ge", recall >= recall_ge, recall, recall_ge)

    for fmt in required_formats:
        fmt_key = str(fmt).strip().lower()
        row = per_format.get(fmt_key)
        if not isinstance(row, Mapping):
            _check(f"per_format.{fmt_key}.present", False, None, "present")
            continue
        parse_success_rate = float(row.get("parse_success_rate", 0.0))
        _check(
            f"per_format.{fmt_key}.parse_success_rate_ge",
            parse_success_rate >= per_format_parse_success_ge,
            parse_success_rate,
            per_format_parse_success_ge,
        )

    return {"passed": all(bool(x["passed"]) for x in checks), "checks": checks}


def _load_baseline_report(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline report must be a JSON object")
    return payload


def _build_regression_block(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    cur_summary = current.get("summary", {}) if isinstance(current.get("summary"), Mapping) else {}
    base_summary = baseline.get("summary", {}) if isinstance(baseline.get("summary"), Mapping) else {}
    keys = ("precision", "recall", "attack_off_rate", "benign_off_rate", "parse_success_rate")
    summary_delta = {
        k: float(cur_summary.get(k, 0.0)) - float(base_summary.get(k, 0.0))
        for k in keys
    }
    per_format_delta: Dict[str, Dict[str, float]] = {}
    cur_pf = current.get("per_format", {}) if isinstance(current.get("per_format"), Mapping) else {}
    base_pf = baseline.get("per_format", {}) if isinstance(baseline.get("per_format"), Mapping) else {}
    for fmt in sorted(set(list(cur_pf.keys()) + list(base_pf.keys()))):
        row_cur = cur_pf.get(fmt, {}) if isinstance(cur_pf.get(fmt), Mapping) else {}
        row_base = base_pf.get(fmt, {}) if isinstance(base_pf.get(fmt), Mapping) else {}
        per_format_delta[str(fmt)] = {
            k: float(row_cur.get(k, 0.0)) - float(row_base.get(k, 0.0))
            for k in keys
        }
    return {"summary_delta": summary_delta, "per_format_delta": per_format_delta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate attachment ingestion FP/FN metrics by format.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--manifest", default="tests/data/attachment_eval/manifest.jsonl")
    parser.add_argument("--artifacts-root", default="artifacts/attachment_eval")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--require-gate", action="store_true")
    parser.add_argument("--gate-attack-off-rate-ge", type=float, default=0.60)
    parser.add_argument("--gate-benign-off-rate-le", type=float, default=0.02)
    parser.add_argument("--gate-precision-ge", type=float, default=0.80)
    parser.add_argument("--gate-recall-ge", type=float, default=0.60)
    parser.add_argument("--gate-per-format-parse-success-ge", type=float, default=0.90)
    parser.add_argument("--gate-required-formats", default="pdf,docx,html,zip")
    parser.add_argument("--gate-include-deferred", action="store_true")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved
    manifest_path = Path(args.manifest).resolve()
    rows = _load_manifest(manifest_path)
    projector = build_projector(cfg)

    retriever_cfg = cfg.get("retriever", {}) or {}
    sqlite_cfg = retriever_cfg.get("sqlite_fts", {}) or {}
    attachment_cfg = sqlite_cfg.get("attachments", {}) if isinstance(sqlite_cfg.get("attachments", {}), dict) else {}

    eval_report = evaluate_attachment_manifest(
        manifest_rows=rows,
        manifest_dir=manifest_path.parent,
        projector=projector,
        attachment_cfg=attachment_cfg,
        use_recommended_verdict=True,
    )
    required_formats = [x.strip().lower() for x in str(args.gate_required_formats).split(",") if x.strip()]
    gate_summary_key = "summary" if bool(args.gate_include_deferred) else "summary_core"
    gate = evaluate_gate(
        summary=eval_report[gate_summary_key],
        summary_label=gate_summary_key,
        per_format=eval_report["per_format"],
        required_formats=required_formats,
        attack_off_rate_ge=float(args.gate_attack_off_rate_ge),
        benign_off_rate_le=float(args.gate_benign_off_rate_le),
        precision_ge=float(args.gate_precision_ge),
        recall_ge=float(args.gate_recall_ge),
        per_format_parse_success_ge=float(args.gate_per_format_parse_success_ge),
    )

    baseline_block: Dict[str, Any] | None = None
    if args.baseline_report:
        baseline_path = Path(str(args.baseline_report)).resolve()
        baseline_payload = _load_baseline_report(baseline_path)
        baseline_block = {
            "path": str(baseline_path),
            "seed_match": int(baseline_payload.get("seed", -1)) == int(args.seed),
            **_build_regression_block(eval_report, baseline_payload),
        }

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"attachment_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = Path(args.artifacts_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    payload = {
        "run_id": run_id,
        "status": "ok" if gate["passed"] else "gate_failed",
        "profile": args.profile,
        "seed": int(args.seed),
        "manifest": str(manifest_path),
        "weekly_regression": bool(args.weekly_regression),
        "gate_scope": "all" if bool(args.gate_include_deferred) else "core_excluding_deferred",
        **eval_report,
        "gate": gate,
        "baseline_compare": baseline_block,
        "artifacts": {"report_json": str(report_path.resolve())},
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if bool(args.require_gate) and not bool(gate["passed"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
