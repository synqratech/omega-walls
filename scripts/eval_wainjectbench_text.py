from __future__ import annotations

import argparse
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
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.interfaces.contracts_v1 import ContentItem, OmegaState, WALLS_V1
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector


WAINJECT_SOURCE_URL = "https://github.com/Norrrrrrr-lyn/WAInjectBench"
WAINJECT_PAPER_URL = "https://arxiv.org/abs/2510.01354"
WAINJECT_SOURCE_DATE_UTC = "2026-03-08"


@dataclass(frozen=True)
class WATextSample:
    sample_id: str
    label: int
    source_file: str
    text: str


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "attack_total": int(tp + fn),
        "benign_total": int(tn + fp),
        "attack_off_rate": _safe_div(tp, tp + fn),
        "benign_off_rate": _safe_div(fp, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
    }


def _load_jsonl_rows(path: Path) -> List[Mapping[str, Any]]:
    out: List[Mapping[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, Mapping):
            out.append(obj)
    return out


def load_wainject_text(root: Path) -> List[WATextSample]:
    samples: List[WATextSample] = []
    for split_dir, label in (("benign", 0), ("malicious", 1)):
        base = root / split_dir
        if not base.exists():
            continue
        for file in sorted(base.glob("*.jsonl")):
            rows = _load_jsonl_rows(file)
            for idx, row in enumerate(rows):
                text = str(row.get("text", row.get("content", ""))).strip()
                if not text:
                    continue
                samples.append(
                    WATextSample(
                        sample_id=f"{split_dir}:{file.name}:{idx:06d}",
                        label=int(label),
                        source_file=file.name,
                        text=text,
                    )
                )
    return samples


def evaluate_wainject_rows(
    *,
    rows: Sequence[WATextSample],
    projector: Any,
    omega_core: OmegaCoreV1,
    off_policy: OffPolicyV1,
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    per_file_counts: Dict[str, Dict[str, int]] = {}
    eval_rows: List[Dict[str, Any]] = []

    for row in rows:
        item = ContentItem(
            doc_id=f"wainject:{row.sample_id}",
            source_id="wainject_text",
            source_type="other",
            trust="untrusted",
            text=row.text,
        )
        proj = projector.project(item)
        state = OmegaState(session_id=f"wainject:{row.sample_id}", m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        step_result = omega_core.step(state=state, items=[item], projections=[proj])
        decision = off_policy.select_actions(step_result=step_result, items=[item])
        pred = bool(decision.off)

        if row.source_file not in per_file_counts:
            per_file_counts[row.source_file] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        cur = per_file_counts[row.source_file]

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
                "label": int(row.label),
                "source_file": row.source_file,
                "pred_attack": pred,
                "off": bool(decision.off),
                "severity": str(decision.severity),
                "v_sum": float(np.sum(proj.v)),
            }
        )

    per_file = {k: _metrics(v["tp"], v["fp"], v["tn"], v["fn"]) for k, v in sorted(per_file_counts.items())}
    return {"summary": _metrics(tp, fp, tn, fn), "per_file": per_file, "rows": eval_rows}


def _latest_baseline_report(artifacts_root: Path) -> Optional[Path]:
    if not artifacts_root.exists():
        return None
    candidates = sorted([p / "report.json" for p in artifacts_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for p in candidates:
        if p.exists():
            return p
    return None


def _baseline_compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("attack_off_rate", "benign_off_rate", "precision", "recall")
    cur = current.get("summary", {}) if isinstance(current.get("summary"), Mapping) else {}
    base = baseline.get("summary", {}) if isinstance(baseline.get("summary"), Mapping) else {}
    delta = {k: float(cur.get(k, 0.0)) - float(base.get(k, 0.0)) for k in keys}
    return {"summary_delta": delta}


def _write_external_refs_json(path: Path) -> None:
    payload = {
        "benchmark": "WAInjectBench",
        "source_date": WAINJECT_SOURCE_DATE_UTC,
        "sources": [
            {
                "url": WAINJECT_SOURCE_URL,
                "type": "benchmark_repo",
                "note": "official dataset repository",
            },
            {
                "url": WAINJECT_PAPER_URL,
                "type": "paper",
                "note": "official benchmark paper",
            },
        ],
        "metric_mapping": {
            "omega_metrics": ["summary.attack_off_rate", "summary.benign_off_rate", "summary.precision", "summary.recall"],
            "external_baseline_table_available": False,
            "reason": "no benchmark-maintainer leaderboard table with detector scores in public source card/readme",
        },
        "comparability_status": "non_comparable",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Omega on WAInjectBench text subset and emit comparability metadata.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--root", default="data/WAInjectBench/text")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap for deterministic sampled evaluation; 0 means full set.")
    parser.add_argument("--artifacts-root", default="artifacts/wainject_eval")
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    root = (ROOT / str(args.root)).resolve()
    samples = load_wainject_text(root)
    if int(args.max_samples) > 0 and len(samples) > int(args.max_samples):
        idx = np.random.RandomState(int(args.seed)).choice(len(samples), size=int(args.max_samples), replace=False)
        selected = [samples[int(i)] for i in sorted(int(x) for x in idx)]
        samples = selected
    dataset_ready = len(samples) > 0

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved

    eval_block: Dict[str, Any] = {"summary": _metrics(0, 0, 0, 0), "per_file": {}, "rows": []}
    if dataset_ready:
        projector = build_projector(cfg)
        omega_core = OmegaCoreV1(omega_params_from_config(cfg))
        off_policy = OffPolicyV1(cfg)
        eval_block = evaluate_wainject_rows(rows=samples, projector=projector, omega_core=omega_core, off_policy=off_policy)

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"wainject_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in eval_block["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    refs_global = ROOT / "artifacts" / "external_refs" / "wainjectbench_refs.json"
    refs_run = out_dir / "wainjectbench_refs.json"
    _write_external_refs_json(refs_global)
    _write_external_refs_json(refs_run)

    baseline_compare = None
    baseline_path = None
    if args.baseline_report:
        baseline_path = (ROOT / str(args.baseline_report)).resolve()
    elif bool(args.weekly_regression):
        latest = _latest_baseline_report((ROOT / str(args.artifacts_root)).resolve())
        if latest is not None and latest.parent.name != run_id:
            baseline_path = latest
    if baseline_path and baseline_path.exists():
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_compare = {"path": str(baseline_path), **_baseline_compare(eval_block, baseline)}

    payload = {
        "run_id": run_id,
        "status": "ok" if dataset_ready else "dataset_not_ready",
        "profile": args.profile,
        "seed": int(args.seed),
        "root": str(root),
        "dataset_ready": dataset_ready,
        "samples_total": int(len(samples)),
        "summary": eval_block["summary"],
        "per_file": eval_block["per_file"],
        "comparability_status": "partial_comparison" if dataset_ready else "non_comparable",
        "external_benchmark": {
            "name": "WAInjectBench (text subset)",
            "source_url": WAINJECT_SOURCE_URL,
            "paper_url": WAINJECT_PAPER_URL,
            "source_date": WAINJECT_SOURCE_DATE_UTC,
            "metric_mapping": {
                "omega_metrics": ["summary.attack_off_rate", "summary.benign_off_rate", "summary.precision", "summary.recall"],
                "external_baseline_table_available": False,
            },
            "comparability_note": "No benchmark-maintainer detector leaderboard table found in official source card/readme; reported as partial only.",
            "evidence_bar": "benchmark-maintainer only",
        },
        "baseline_compare": baseline_compare,
        "artifacts": {
            "report_json": str(report_path),
            "rows_jsonl": str(rows_path),
            "external_refs_global_json": str(refs_global),
            "external_refs_run_json": str(refs_run),
        },
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
