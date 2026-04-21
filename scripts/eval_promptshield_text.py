from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OmegaState, WALLS_V1
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector


PROMPTSHIELD_SOURCE_DATE_UTC = "2026-03-12"


@dataclass(frozen=True)
class PromptShieldSample:
    sample_id: str
    label: int
    split: str
    source_file: str
    text: str


@dataclass(frozen=True)
class PromptShieldLoadStats:
    split: str
    file_exists: bool
    rows_seen: int
    rows_loaded: int
    rows_dropped_non_mapping: int
    rows_dropped_empty_text: int
    rows_dropped_missing_label: int
    rows_dropped_too_long: int
    samples_built: int
    attack_samples_built: int
    benign_samples_built: int


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "attack_total": int(tp + fn),
        "benign_total": int(tn + fp),
        "attack_off_rate": float(tpr),
        "benign_off_rate": _safe_div(fp, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "recall": float(tpr),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
    }


def _normalize_label(raw: Any) -> Optional[int]:
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, (int, float)):
        return 1 if int(raw) == 1 else 0
    txt = str(raw).strip().lower()
    if txt in {"1", "attack", "malicious", "unsafe", "true"}:
        return 1
    if txt in {"0", "benign", "safe", "false", "clean"}:
        return 0
    return None


def _extract_text(row: Mapping[str, Any]) -> str:
    for key in ("prompt", "text", "input", "content", "user_input"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def load_promptshield_split_with_stats(root: Path, split: str) -> tuple[List[PromptShieldSample], PromptShieldLoadStats]:
    split_name = str(split).strip().lower()
    path = root / f"{split_name}.json"
    if not path.exists():
        stats = PromptShieldLoadStats(
            split=split_name,
            file_exists=False,
            rows_seen=0,
            rows_loaded=0,
            rows_dropped_non_mapping=0,
            rows_dropped_empty_text=0,
            rows_dropped_missing_label=0,
            rows_dropped_too_long=0,
            samples_built=0,
            attack_samples_built=0,
            benign_samples_built=0,
        )
        return [], stats

    raw_obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_obj, list):
        stats = PromptShieldLoadStats(
            split=split_name,
            file_exists=True,
            rows_seen=0,
            rows_loaded=0,
            rows_dropped_non_mapping=0,
            rows_dropped_empty_text=0,
            rows_dropped_missing_label=0,
            rows_dropped_too_long=0,
            samples_built=0,
            attack_samples_built=0,
            benign_samples_built=0,
        )
        return [], stats

    rows_seen = int(len(raw_obj))
    rows_loaded = 0
    rows_dropped_non_mapping = 0
    rows_dropped_empty_text = 0
    rows_dropped_missing_label = 0
    attack_samples_built = 0
    benign_samples_built = 0
    out: List[PromptShieldSample] = []

    for idx, raw in enumerate(raw_obj):
        if not isinstance(raw, Mapping):
            rows_dropped_non_mapping += 1
            continue
        rows_loaded += 1
        text = _extract_text(raw)
        if not text:
            rows_dropped_empty_text += 1
            continue
        label = _normalize_label(raw.get("label"))
        if label is None:
            rows_dropped_missing_label += 1
            continue
        sample = PromptShieldSample(
            sample_id=f"{split_name}:{idx:06d}",
            label=int(label),
            split=split_name,
            source_file=f"{split_name}.json",
            text=text,
        )
        out.append(sample)
        if int(label) == 1:
            attack_samples_built += 1
        else:
            benign_samples_built += 1

    stats = PromptShieldLoadStats(
        split=split_name,
        file_exists=True,
        rows_seen=rows_seen,
        rows_loaded=rows_loaded,
        rows_dropped_non_mapping=rows_dropped_non_mapping,
        rows_dropped_empty_text=rows_dropped_empty_text,
        rows_dropped_missing_label=rows_dropped_missing_label,
        rows_dropped_too_long=0,
        samples_built=len(out),
        attack_samples_built=attack_samples_built,
        benign_samples_built=benign_samples_built,
    )
    return out, stats


def _stratified_cap_samples(
    samples: Sequence[PromptShieldSample],
    *,
    max_samples: int,
    seed: int,
) -> tuple[List[PromptShieldSample], Dict[str, Any]]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return list(samples), {"strategy": "full", "requested_max_samples": int(max_samples), "selected_total": int(len(samples))}

    grouped: Dict[int, List[PromptShieldSample]] = {0: [], 1: []}
    for row in samples:
        grouped[int(row.label)].append(row)

    rng = np.random.RandomState(int(seed))
    floors: Dict[int, int] = {}
    remainders: List[tuple[float, int]] = []
    total = float(len(samples))
    for lbl in (0, 1):
        vals = grouped[lbl]
        perm = rng.permutation(len(vals))
        grouped[lbl] = [vals[int(i)] for i in perm]
        quota_raw = float(max_samples) * float(len(vals)) / total
        quota_floor = min(len(vals), int(np.floor(quota_raw)))
        floors[lbl] = quota_floor
        remainders.append((quota_raw - float(quota_floor), lbl))

    used = int(sum(floors.values()))
    remain = int(max_samples - used)
    for _, lbl in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if remain <= 0:
            break
        if floors[lbl] < len(grouped[lbl]):
            floors[lbl] += 1
            remain -= 1

    selected = grouped[0][: floors[0]] + grouped[1][: floors[1]]
    selected.sort(key=lambda x: x.sample_id)
    return selected, {
        "strategy": "stratified_label",
        "requested_max_samples": int(max_samples),
        "selected_total": int(len(selected)),
        "by_label": {
            "0": {"available": int(len(grouped[0])), "selected": int(floors[0])},
            "1": {"available": int(len(grouped[1])), "selected": int(floors[1])},
        },
    }


def _filter_samples_by_max_chars(
    samples: Sequence[PromptShieldSample],
    *,
    max_text_chars: int,
) -> tuple[List[PromptShieldSample], int]:
    if int(max_text_chars) <= 0:
        return list(samples), 0
    out = [s for s in samples if len(s.text) <= int(max_text_chars)]
    dropped = int(len(samples) - len(out))
    return out, dropped


def _deterministic_shuffle_samples(
    samples: Sequence[PromptShieldSample],
    *,
    seed: int,
) -> List[PromptShieldSample]:
    rows = list(samples)
    if len(rows) <= 1:
        return rows
    rng = np.random.RandomState(int(seed))
    perm = rng.permutation(len(rows))
    return [rows[int(i)] for i in perm]


def evaluate_promptshield_rows(
    *,
    rows: Sequence[PromptShieldSample],
    projector: Any,
    omega_core: OmegaCoreV1,
    off_policy: OffPolicyV1,
    max_seconds: float = 0.0,
) -> Dict[str, Any]:
    started = time.monotonic()
    tp = fp = tn = fn = 0
    per_source_counts: Dict[str, Dict[str, int]] = {}
    eval_rows: List[Dict[str, Any]] = []
    processed_total = 0
    stop_reason: Optional[str] = None

    for row in rows:
        if float(max_seconds) > 0.0 and processed_total > 0:
            if (time.monotonic() - started) >= float(max_seconds):
                stop_reason = "time_budget_reached"
                break
        item = ContentItem(
            doc_id=f"promptshield:{row.sample_id}",
            source_id="promptshield_text",
            source_type="other",
            trust="untrusted",
            text=row.text,
        )
        proj = projector.project(item)
        state = OmegaState(session_id=f"promptshield:{row.sample_id}", m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        step_result = omega_core.step(state=state, items=[item], projections=[proj])
        decision = off_policy.select_actions(step_result=step_result, items=[item])
        pred = bool(decision.off)

        per_source_counts.setdefault(row.source_file, {"tp": 0, "fp": 0, "tn": 0, "fn": 0})
        cur = per_source_counts[row.source_file]

        if row.label == 1 and pred:
            tp += 1
            cur["tp"] += 1
        elif row.label == 1 and not pred:
            fn += 1
            cur["fn"] += 1
        elif row.label == 0 and pred:
            fp += 1
            cur["fp"] += 1
        else:
            tn += 1
            cur["tn"] += 1

        eval_rows.append(
            {
                "id": row.sample_id,
                "split": row.split,
                "source_file": row.source_file,
                "label": int(row.label),
                "pred_attack": bool(pred),
                "off": bool(decision.off),
                "severity": str(decision.severity),
                "v_sum": float(np.sum(proj.v)),
                "text": row.text,
            }
        )
        processed_total += 1

    per_source = {k: _metrics(v["tp"], v["fp"], v["tn"], v["fn"]) for k, v in sorted(per_source_counts.items())}
    elapsed = float(max(0.0, time.monotonic() - started))
    sps = (float(processed_total) / elapsed) if elapsed > 0.0 else 0.0
    return {
        "summary": _metrics(tp, fp, tn, fn),
        "per_source": per_source,
        "rows": eval_rows,
        "processed_total": int(processed_total),
        "selected_total": int(len(rows)),
        "stopped_early": bool(processed_total < len(rows)),
        "stop_reason": stop_reason,
        "runtime": {
            "elapsed_sec": elapsed,
            "samples_per_sec": sps,
            "max_seconds": float(max_seconds),
        },
    }


def _baseline_compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("attack_off_rate", "benign_off_rate", "precision", "recall", "balanced_accuracy")
    cur = current.get("summary", {}) if isinstance(current.get("summary"), Mapping) else {}
    base = baseline.get("summary", {}) if isinstance(baseline.get("summary"), Mapping) else {}
    delta = {k: float(cur.get(k, 0.0)) - float(base.get(k, 0.0)) for k in keys}
    return {"summary_delta": delta}


def _resolve_comparability_for_promptshield(
    *,
    dataset_ready: bool,
    max_samples: int,
    selected_total: int,
    evaluated_total: int,
    stats: PromptShieldLoadStats,
) -> tuple[str, str]:
    if not bool(dataset_ready):
        return "non_comparable", "dataset_not_ready"
    if int(stats.attack_samples_built) <= 0 or int(stats.benign_samples_built) <= 0:
        return "non_comparable", "missing_attack_or_benign_labels"
    if int(evaluated_total) < int(selected_total):
        return "non_comparable", "time_budget_reached"
    if int(max_samples) > 0 and int(selected_total) < int(stats.samples_built):
        return "non_comparable", "subsampled_run_max_samples"
    return "non_comparable", "no_benchmark_maintainer_detector_leaderboard"


def _cluster_promptshield_text(text: str) -> str:
    t = str(text).lower()
    grammar_markers = (
        "rewrite",
        "rephrase",
        "more concise",
        "grammar",
        "grammatical",
        "proofread",
        "edit",
        "polite email",
    )
    soft_directive_markers = (
        "i want you",
        "your task is",
        "answer with",
        "respond with",
        "reply with",
        "your answer must",
        "answer according to",
        "could you",
        "can you",
        "please provide",
    )
    task_like_markers = (
        "given the following text",
        "are the following two sentences",
        "write the sentiment",
        "what is the sentiment",
        "classification",
        "extract the",
        "create ",
        "write ",
        "generate ",
        "describe ",
        "name ",
        "summarize ",
        "manual",
        "guide",
        "essay",
        "function",
        "program",
    )
    override_cues = (
        "ignore previous instructions",
        "disregard the above instructions",
        "override previous rules",
        "forget previous instructions",
        "print above prompt",
        "return embeddings",
        "system prompt",
    )

    if any(x in t for x in grammar_markers):
        return "grammar_edit_wrappers"
    if any(x in t for x in soft_directive_markers) and not any(x in t for x in override_cues):
        return "soft_directive_no_override"
    if any(x in t for x in task_like_markers):
        return "task_like_eval"
    return "task_like_eval"


def _cluster_promptshield_audit_flags(text: str) -> Dict[str, Any]:
    t = str(text).lower()
    weak_markers = ("previous", "prior", "above", "skip", "earlier", "before")
    override_cues = (
        "ignore previous instructions",
        "disregard the above instructions",
        "override previous rules",
        "forget previous instructions",
        "print above prompt",
        "return embeddings",
        "system prompt",
    )
    return {
        "weak_marker_hit": bool(any(x in t for x in weak_markers)),
        "override_cue_hit": bool(any(x in t for x in override_cues)),
    }


def _target_wall_from_text(text: str) -> str:
    t = str(text).lower()
    if any(
        x in t
        for x in (
            "embedding",
            "print above prompt",
            "system prompt",
            "prompt text",
            "secret key",
            "only print key",
            "only output key",
            "output only the key",
            "print only the key",
        )
    ):
        return "secret_exfiltration"
    if any(x in t for x in ("open a new tab", "click", "upvote", "subscribe", "fill out", "do not ask user", "retry")):
        return "tool_or_action_abuse"
    return "override_instructions"


def build_iter1_cases_from_eval_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_fp: int = 64,
    max_fn: int = 64,
) -> List[Dict[str, Any]]:
    fp_rows = [r for r in rows if int(r.get("label", 0)) == 0 and bool(r.get("pred_attack", False))]
    fn_rows = [r for r in rows if int(r.get("label", 0)) == 1 and not bool(r.get("pred_attack", False))]
    fp_rows = sorted(fp_rows, key=lambda r: str(r.get("id", "")))[: int(max_fp)]
    fn_rows = sorted(fn_rows, key=lambda r: str(r.get("id", "")))[: int(max_fn)]

    out: List[Dict[str, Any]] = []
    for i, row in enumerate(fp_rows, start=1):
        text = str(row.get("text", ""))
        out.append(
            {
                "id": f"PSI1-FP-{i:04d}",
                "source": "promptshield_validation",
                "origin_sample_id": str(row.get("id", "")),
                "cluster": _cluster_promptshield_text(text),
                "cluster_audit": _cluster_promptshield_audit_flags(text),
                "text": text,
                "expected_mode": "block",
                "expected_all_zero": True,
                "expected_nonzero": [],
            }
        )
    for i, row in enumerate(fn_rows, start=1):
        text = str(row.get("text", ""))
        out.append(
            {
                "id": f"PSI1-FN-{i:04d}",
                "source": "promptshield_validation",
                "origin_sample_id": str(row.get("id", "")),
                "cluster": _cluster_promptshield_text(text),
                "cluster_audit": _cluster_promptshield_audit_flags(text),
                "text": text,
                "expected_mode": "block",
                "expected_all_zero": False,
                "expected_nonzero": [_target_wall_from_text(text)],
            }
        )
    return out


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Omega on PromptShield split and emit non-comparable benchmark report.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--root", default="data/PromptShield")
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-text-chars", type=int, default=0, help="Drop rows with text length above this value; 0 disables.")
    parser.add_argument("--max-seconds", type=float, default=0.0, help="Hard time budget for evaluation loop; 0 disables.")
    parser.add_argument("--artifacts-root", default="artifacts/promptshield_eval")
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--export-iter1-cases-out", default=None)
    parser.add_argument("--export-iter1-max-fp", type=int, default=64)
    parser.add_argument("--export-iter1-max-fn", type=int, default=64)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    root = (ROOT / str(args.root)).resolve()
    samples, stats = load_promptshield_split_with_stats(root, str(args.split))
    samples, dropped_too_long = _filter_samples_by_max_chars(samples, max_text_chars=int(args.max_text_chars))
    stats = PromptShieldLoadStats(
        split=stats.split,
        file_exists=stats.file_exists,
        rows_seen=stats.rows_seen,
        rows_loaded=stats.rows_loaded,
        rows_dropped_non_mapping=stats.rows_dropped_non_mapping,
        rows_dropped_empty_text=stats.rows_dropped_empty_text,
        rows_dropped_missing_label=stats.rows_dropped_missing_label,
        rows_dropped_too_long=int(dropped_too_long),
        samples_built=len(samples),
        attack_samples_built=sum(1 for s in samples if int(s.label) == 1),
        benign_samples_built=sum(1 for s in samples if int(s.label) == 0),
    )
    sampling_info: Dict[str, Any] = {"strategy": "full", "requested_max_samples": int(args.max_samples), "selected_total": int(len(samples))}
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        samples, sampling_info = _stratified_cap_samples(samples, max_samples=int(args.max_samples), seed=int(args.seed))
    if float(args.max_seconds) > 0.0 and len(samples) > 1:
        samples = _deterministic_shuffle_samples(samples, seed=int(args.seed))
        sampling_info["evaluation_order"] = "deterministic_shuffle_seeded"
    else:
        sampling_info["evaluation_order"] = "stable"
    dataset_ready = len(samples) > 0
    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved

    eval_block: Dict[str, Any] = {"summary": _metrics(0, 0, 0, 0), "per_source": {}, "rows": []}
    if dataset_ready:
        projector = build_projector(cfg)
        omega_core = OmegaCoreV1(omega_params_from_config(cfg))
        off_policy = OffPolicyV1(cfg)
        eval_block = evaluate_promptshield_rows(
            rows=samples,
            projector=projector,
            omega_core=omega_core,
            off_policy=off_policy,
            max_seconds=float(args.max_seconds),
        )
    comparability_status, comparability_reason = _resolve_comparability_for_promptshield(
        dataset_ready=dataset_ready,
        max_samples=int(args.max_samples),
        selected_total=int(eval_block.get("selected_total", len(samples))),
        evaluated_total=int(eval_block.get("processed_total", 0)),
        stats=stats,
    )

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"promptshield_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    rows_path = out_dir / "rows.jsonl"
    _write_jsonl(rows_path, eval_block["rows"])

    iter1_cases_path: Optional[Path] = None
    iter1_cases_count = 0
    if args.export_iter1_cases_out:
        iter1_cases_path = (ROOT / str(args.export_iter1_cases_out)).resolve()
        cases = build_iter1_cases_from_eval_rows(
            eval_block["rows"],
            max_fp=int(args.export_iter1_max_fp),
            max_fn=int(args.export_iter1_max_fn),
        )
        _write_jsonl(iter1_cases_path, cases)
        iter1_cases_count = len(cases)

    baseline_compare = None
    if args.baseline_report:
        baseline_path = (ROOT / str(args.baseline_report)).resolve()
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_compare = {"path": str(baseline_path), **_baseline_compare(eval_block, baseline)}

    payload = {
        "run_id": run_id,
        "status": "ok" if dataset_ready else "dataset_not_ready",
        "profile": args.profile,
        "seed": int(args.seed),
        "root": str(root),
        "split": str(args.split),
        "dataset_ready": bool(dataset_ready),
        "samples_total": int(len(samples)),
        "samples_evaluated_total": int(eval_block.get("processed_total", 0)),
        "evaluation_truncated": bool(eval_block.get("stopped_early", False)),
        "dataset_rows": {
            "file_exists": bool(stats.file_exists),
            "rows_seen": int(stats.rows_seen),
            "rows_loaded": int(stats.rows_loaded),
            "rows_dropped_non_mapping": int(stats.rows_dropped_non_mapping),
            "rows_dropped_empty_text": int(stats.rows_dropped_empty_text),
            "rows_dropped_missing_label": int(stats.rows_dropped_missing_label),
            "rows_dropped_too_long": int(stats.rows_dropped_too_long),
            "samples_built": int(stats.samples_built),
        },
        "sampling": sampling_info,
        "comparability_status": comparability_status,
        "comparability_reason": comparability_reason,
        "evaluation_mode": {
            "runtime": "stateless_per_sample",
            "session_runtime": False,
            "note": "PromptShield rows are evaluated as independent text samples.",
        },
        "summary": eval_block["summary"],
        "per_split": {str(args.split): eval_block["summary"]},
        "per_source": eval_block["per_source"],
        "external_benchmark": {
            "name": "PromptShield",
            "source_url": None,
            "source_date": PROMPTSHIELD_SOURCE_DATE_UTC,
            "metric_mapping": {
                "omega_metrics": [
                    "summary.attack_off_rate",
                    "summary.benign_off_rate",
                    "summary.precision",
                    "summary.recall",
                    "summary.balanced_accuracy",
                ],
                "external_baseline_table_available": False,
            },
            "comparability_note": "No benchmark-maintainer detector leaderboard table attached to this local PromptShield eval protocol.",
            "evidence_bar": "benchmark-maintainer only",
        },
        "baseline_compare": baseline_compare,
        "runtime": eval_block.get("runtime", {}),
        "artifacts": {
            "report_json": str(report_path),
            "rows_jsonl": str(rows_path),
            "iter1_cases_jsonl": str(iter1_cases_path) if iter1_cases_path else None,
            "iter1_cases_count": int(iter1_cases_count),
        },
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
