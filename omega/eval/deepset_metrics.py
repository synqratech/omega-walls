"""Deepset benchmark metrics for Omega security evaluation."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from omega.core.omega_core import OmegaCoreV1
from omega.eval.deepset_adapter import build_deepset_samples
from omega.interfaces.contracts_v1 import ContentItem, OmegaState, WALLS_V1
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2


def _rate(num: int, den: int) -> float:
    if den <= 0:
        return 0.0
    return float(num) / float(den)


def _safe_precision(tp: int, fp: int) -> float:
    return _rate(tp, tp + fp)


def _safe_recall(tp: int, fn: int) -> float:
    return _rate(tp, tp + fn)


def _safe_f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def _script_bucket(text: str) -> str:
    return "ascii" if all(ord(ch) < 128 for ch in str(text)) else "non_ascii"


def _new_stratum_state() -> Dict[str, Any]:
    return {
        "samples": {"total": 0, "attack_total": 0, "benign_total": 0},
        "confusion": {"tp": 0, "fp": 0, "tn": 0, "fn": 0},
        "attack_wall_any": 0,
        "per_wall_activation": {w: 0 for w in WALLS_V1},
    }


def _finalize_stratum(state: Dict[str, Any]) -> Dict[str, Any]:
    samples = state["samples"]
    confusion = state["confusion"]
    attack_total = int(samples["attack_total"])
    benign_total = int(samples["benign_total"])
    tp = int(confusion["tp"])
    fp = int(confusion["fp"])
    tn = int(confusion["tn"])
    fn = int(confusion["fn"])
    precision = _safe_precision(tp, fp)
    recall = _safe_recall(tp, fn)
    f1 = _safe_f1(precision, recall)
    tpr = recall
    tnr = _rate(tn, tn + fp)
    return {
        "samples": {
            "total": int(samples["total"]),
            "attack_total": attack_total,
            "benign_total": benign_total,
        },
        "metrics": {
            "attack_off_rate": _rate(tp, attack_total),
            "benign_off_rate": _rate(fp, benign_total),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": (tpr + tnr) / 2.0,
            "coverage_wall_any_attack": _rate(int(state["attack_wall_any"]), attack_total),
            "per_wall_activation_rate": {
                wall: _rate(int(count), attack_total)
                for wall, count in state["per_wall_activation"].items()
            },
        },
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
    }


def evaluate_deepset(
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
) -> Dict[str, Any]:
    bundle = build_deepset_samples(
        benchmark_root=benchmark_root,
        split=split,
        mode=mode,
        max_samples=max_samples,
        seed=seed,
        label_attack_value=label_attack_value,
    )
    idx = {w: i for i, w in enumerate(WALLS_V1)}

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    attack_wall_any = 0
    per_wall_activation = {w: 0 for w in WALLS_V1}
    attack_sum_p: List[float] = []
    benign_sum_p: List[float] = []
    by_script: Dict[str, Dict[str, Any]] = {
        "ascii": _new_stratum_state(),
        "non_ascii": _new_stratum_state(),
    }

    for sample in bundle.samples:
        bucket = _script_bucket(sample.text)
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
        sum_p = float(np.sum(step_result.p))
        if sample.is_attack:
            attack_sum_p.append(sum_p)
        else:
            benign_sum_p.append(sum_p)
        group = by_script[bucket]
        group["samples"]["total"] += 1
        if sample.is_attack:
            group["samples"]["attack_total"] += 1
        else:
            group["samples"]["benign_total"] += 1

        if sample.is_attack and pred_attack:
            tp += 1
            group["confusion"]["tp"] += 1
        elif sample.is_attack and not pred_attack:
            fn += 1
            group["confusion"]["fn"] += 1
        elif (not sample.is_attack) and pred_attack:
            fp += 1
            group["confusion"]["fp"] += 1
        else:
            tn += 1
            group["confusion"]["tn"] += 1

        if sample.is_attack:
            if float(projection.v.sum()) > 0.0:
                attack_wall_any += 1
                group["attack_wall_any"] += 1
            for wall in WALLS_V1:
                if float(projection.v[idx[wall]]) > 0.0:
                    per_wall_activation[wall] += 1
                    group["per_wall_activation"][wall] += 1

    attack_total = bundle.attack_selected
    benign_total = bundle.benign_selected
    total = attack_total + benign_total

    precision = _safe_precision(tp, fp)
    recall = _safe_recall(tp, fn)
    f1 = _safe_f1(precision, recall)
    tpr = recall
    tnr = _rate(tn, tn + fp)

    return {
        "status": "ok",
        "mode": mode,
        "split": split,
        "seed": seed,
        "label_attack_value": int(label_attack_value),
        "samples": {
            "total": total,
            "attack_total": attack_total,
            "benign_total": benign_total,
            "selected_rows": bundle.selected_rows,
            "total_rows": bundle.total_rows,
            "dropped_empty_text_rows": bundle.dropped_empty_text_rows,
        },
        "metrics": {
            "attack_off_rate": _rate(tp, attack_total),
            "benign_off_rate": _rate(fp, benign_total),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "balanced_accuracy": (tpr + tnr) / 2.0,
            "coverage_wall_any_attack": _rate(attack_wall_any, attack_total),
            "per_wall_activation_rate": {wall: _rate(count, attack_total) for wall, count in per_wall_activation.items()},
            "attack_p95_sum_p": float(np.percentile(np.array(attack_sum_p, dtype=float), 95)) if attack_sum_p else 0.0,
            "benign_p95_sum_p": float(np.percentile(np.array(benign_sum_p, dtype=float), 95)) if benign_sum_p else 0.0,
        },
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "stratified": {
            "script": {
                "ascii": _finalize_stratum(by_script["ascii"]),
                "non_ascii": _finalize_stratum(by_script["non_ascii"]),
            }
        },
    }


def evaluate_deepset_thresholds(report: Dict[str, Any], *, thresholds: Dict[str, Any]) -> List[str]:
    failures: List[str] = []
    metrics = report.get("metrics", {})

    attack_off_rate = float(metrics.get("attack_off_rate", 0.0))
    if attack_off_rate < float(thresholds.get("attack_off_rate_ge", 0.0)):
        failures.append(f"deepset attack_off_rate < {thresholds.get('attack_off_rate_ge')} ({attack_off_rate:.3f})")

    benign_off_rate = float(metrics.get("benign_off_rate", 1.0))
    if benign_off_rate > float(thresholds.get("benign_off_rate_le", 1.0)):
        failures.append(f"deepset benign_off_rate > {thresholds.get('benign_off_rate_le')} ({benign_off_rate:.3f})")

    f1 = float(metrics.get("f1", 0.0))
    if f1 < float(thresholds.get("f1_ge", 0.0)):
        failures.append(f"deepset f1 < {thresholds.get('f1_ge')} ({f1:.3f})")

    coverage = float(metrics.get("coverage_wall_any_attack", 0.0))
    if coverage < float(thresholds.get("coverage_wall_any_attack_ge", 0.0)):
        failures.append(
            "deepset coverage_wall_any_attack "
            f"< {thresholds.get('coverage_wall_any_attack_ge')} ({coverage:.3f})"
        )

    return failures
