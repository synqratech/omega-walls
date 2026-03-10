from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

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


PINT_SOURCE_URL = "https://github.com/lakeraai/pint-benchmark"
PINT_SCOREBOARD_DATE_UTC = "2026-03-08"
PINT_PUBLIC_BASELINES: List[Dict[str, Any]] = [
    {"system": "Lakera Guard", "pint_score_pct": 95.2204, "evaluation_date_utc": "2025-05-02"},
    {"system": "Aporia Prompt Guard", "pint_score_pct": 90.6354, "evaluation_date_utc": "2025-11-17"},
    {"system": "AWS Prompt Attack Detection", "pint_score_pct": 90.5973, "evaluation_date_utc": "2025-07-10"},
    {"system": "Azure Prompt Shield", "pint_score_pct": 89.9960, "evaluation_date_utc": "2025-08-27"},
    {"system": "ProtectAI Prompt Injection Detector", "pint_score_pct": 81.3897, "evaluation_date_utc": "2025-05-02"},
    {"system": "Llama PromptGuard 2", "pint_score_pct": 77.3294, "evaluation_date_utc": "2025-07-03"},
    {"system": "Google Model Armor", "pint_score_pct": 75.2211, "evaluation_date_utc": "2025-08-27"},
    {"system": "Llama PromptGuard", "pint_score_pct": 65.7136, "evaluation_date_utc": "2025-05-02"},
]


@dataclass(frozen=True)
class PINTSample:
    sample_id: str
    text: str
    label: int
    family: str


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def _normalize_label(raw: Any) -> int:
    if isinstance(raw, bool):
        return 1 if raw else 0
    if isinstance(raw, (int, float)):
        return 1 if int(raw) == 1 else 0
    txt = str(raw).strip().lower()
    if txt in {"1", "attack", "malicious", "true", "unsafe", "injection"}:
        return 1
    if txt in {"0", "benign", "false", "safe", "clean"}:
        return 0
    raise ValueError(f"unsupported PINT label: {raw!r}")


def _extract_text(row: Mapping[str, Any]) -> str:
    for key in ("text", "prompt", "input", "content", "user_input"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_family(row: Mapping[str, Any]) -> str:
    for key in ("family", "category", "attack_type", "type"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return "unknown"


def load_pint_dataset(path: Path) -> List[PINTSample]:
    if not path.exists():
        return []
    raw_obj = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows: Sequence[Any]
    if isinstance(raw_obj, list):
        rows = raw_obj
    elif isinstance(raw_obj, dict):
        for key in ("samples", "rows", "data", "dataset", "benchmark_dataset"):
            val = raw_obj.get(key)
            if isinstance(val, list):
                rows = val
                break
        else:
            rows = []
    else:
        rows = []

    samples: List[PINTSample] = []
    for idx, raw in enumerate(rows):
        if not isinstance(raw, Mapping):
            continue
        text = _extract_text(raw)
        if not text:
            continue
        label_raw = raw.get("label", raw.get("is_attack", raw.get("malicious", raw.get("target", None))))
        if label_raw is None:
            continue
        try:
            label = _normalize_label(label_raw)
        except Exception:
            continue
        sample_id = str(raw.get("id", f"pint-{idx:06d}"))
        samples.append(
            PINTSample(
                sample_id=sample_id,
                text=text,
                label=label,
                family=_extract_family(raw),
            )
        )
    return samples


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
        "attack_off_rate": tpr,
        "benign_off_rate": _safe_div(fp, tn + fp),
        "precision": _safe_div(tp, tp + fp),
        "recall": tpr,
        "balanced_accuracy": (tpr + tnr) / 2.0,
        "pint_score_pct": 100.0 * ((tpr + tnr) / 2.0),
    }


def evaluate_pint_rows(
    *,
    rows: Sequence[PINTSample],
    projector: Any,
    omega_core: OmegaCoreV1,
    off_policy: OffPolicyV1,
) -> Dict[str, Any]:
    tp = fp = tn = fn = 0
    by_family: Dict[str, Dict[str, int]] = {}
    eval_rows: List[Dict[str, Any]] = []

    for row in rows:
        item = ContentItem(
            doc_id=f"pint:{row.sample_id}",
            source_id="pint_benchmark",
            source_type="other",
            trust="untrusted",
            text=row.text,
        )
        proj = projector.project(item)
        state = OmegaState(session_id=f"pint:{row.sample_id}", m=np.zeros(len(WALLS_V1), dtype=float), step=0)
        step_result = omega_core.step(state=state, items=[item], projections=[proj])
        decision = off_policy.select_actions(step_result=step_result, items=[item])
        pred_attack = bool(decision.off)

        fam = row.family
        if fam not in by_family:
            by_family[fam] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        if row.label == 1 and pred_attack:
            tp += 1
            by_family[fam]["tp"] += 1
        elif row.label == 1 and not pred_attack:
            fn += 1
            by_family[fam]["fn"] += 1
        elif row.label == 0 and pred_attack:
            fp += 1
            by_family[fam]["fp"] += 1
        else:
            tn += 1
            by_family[fam]["tn"] += 1

        eval_rows.append(
            {
                "id": row.sample_id,
                "family": row.family,
                "label": int(row.label),
                "pred_attack": pred_attack,
                "off": bool(decision.off),
                "severity": str(decision.severity),
                "v_sum": float(np.sum(proj.v)),
            }
        )

    per_family = {fam: _metrics(v["tp"], v["fp"], v["tn"], v["fn"]) for fam, v in sorted(by_family.items())}
    return {"summary": _metrics(tp, fp, tn, fn), "per_family": per_family, "rows": eval_rows}


def _sort_scoreboard(scoreboard: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [{**dict(x)} for x in scoreboard],
        key=lambda x: float(x.get("pint_score_pct", 0.0)),
        reverse=True,
    )


def _latest_baseline_report(artifacts_root: Path) -> Optional[Path]:
    if not artifacts_root.exists():
        return None
    candidates = sorted([p / "report.json" for p in artifacts_root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for p in candidates:
        if p.exists():
            return p
    return None


def _baseline_compare(current: Mapping[str, Any], baseline: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("pint_score_pct", "attack_off_rate", "benign_off_rate", "precision", "recall")
    cur = current.get("summary", {}) if isinstance(current.get("summary"), Mapping) else {}
    base = baseline.get("summary", {}) if isinstance(baseline.get("summary"), Mapping) else {}
    delta = {k: float(cur.get(k, 0.0)) - float(base.get(k, 0.0)) for k in keys}
    return {"summary_delta": delta}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Omega on PINT-format dataset and compare to public benchmark-maintainer scores.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--dataset", default="data/pint-benchmark/benchmark/data/benchmark_dataset.yaml")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--artifacts-root", default="artifacts/pint_eval")
    parser.add_argument("--weekly-regression", action="store_true")
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--require-dataset", action="store_true")
    args = parser.parse_args()

    np.random.seed(int(args.seed))
    dataset_path = (ROOT / str(args.dataset)).resolve()
    rows = load_pint_dataset(dataset_path)
    dataset_ready = len(rows) > 0

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved

    eval_block: Dict[str, Any] = {
        "summary": _metrics(0, 0, 0, 0),
        "per_family": {},
        "rows": [],
    }
    if dataset_ready:
        projector = build_projector(cfg)
        omega_core = OmegaCoreV1(omega_params_from_config(cfg))
        off_policy = OffPolicyV1(cfg)
        eval_block = evaluate_pint_rows(rows=rows, projector=projector, omega_core=omega_core, off_policy=off_policy)

    now = datetime.now(timezone.utc)
    weekly_tag = f"_w{now.strftime('%G%V')}" if bool(args.weekly_regression) else ""
    run_id = f"pint_eval{weekly_tag}_{_utc_compact_now()}"
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.json"
    rows_path = out_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for row in eval_block["rows"]:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    scoreboard = list(PINT_PUBLIC_BASELINES)
    if dataset_ready:
        scoreboard.append(
            {
                "system": "Omega Walls (this run)",
                "pint_score_pct": float(eval_block["summary"]["pint_score_pct"]),
                "evaluation_date_utc": now.strftime("%Y-%m-%d"),
            }
        )
    ranking = _sort_scoreboard(scoreboard)

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
        "dataset_path": str(dataset_path),
        "dataset_ready": dataset_ready,
        "samples_total": int(len(rows)),
        "summary": eval_block["summary"],
        "per_family": eval_block["per_family"],
        "comparability_status": "direct_comparison" if dataset_ready else "non_comparable",
        "external_benchmark": {
            "name": "PINT",
            "source_url": PINT_SOURCE_URL,
            "source_date": PINT_SCOREBOARD_DATE_UTC,
            "metric_mapping": {
                "benchmark_metric": "PINT score (balanced accuracy, %)",
                "omega_metric": "summary.pint_score_pct",
            },
            "public_scoreboard": _sort_scoreboard(PINT_PUBLIC_BASELINES),
            "rank_with_omega_if_available": ranking,
            "evidence_bar": "benchmark-maintainer only",
        },
        "baseline_compare": baseline_compare,
        "artifacts": {
            "report_json": str(report_path),
            "rows_jsonl": str(rows_path),
        },
    }
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if bool(args.require_dataset) and not dataset_ready:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
