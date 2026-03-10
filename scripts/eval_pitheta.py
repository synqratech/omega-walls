from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.pitheta.eval_gates import GateResult, evaluate_pitheta_gates
from omega.pitheta.dataset_builder import load_pitheta_jsonl
from omega.projector.pitheta_inference import PiThetaInference, PiThetaInferenceConfig


def _extract_json_blob(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("unable to locate JSON object in output")
    return json.loads(text[start : end + 1])


def _run_eval(
    *,
    profile: str,
    checkpoint_dir: str,
    deepset_root: str,
    require_semantic: bool,
    require_calibration: bool,
) -> Dict[str, Any]:
    argv = [
        sys.executable,
        "scripts/run_eval.py",
        "--profile",
        profile,
        "--enforce-whitebox",
        "--whitebox-max-samples",
        "200",
        "--run-deepset",
        "--deepset-benchmark-root",
        deepset_root,
        "--deepset-split",
        "test",
        "--deepset-mode",
        "full",
        "--deepset-seed",
        "41",
        "--projector-mode",
        "pitheta",
        "--pitheta-checkpoint-dir",
        checkpoint_dir,
        "--pitheta-base-model-path",
        "deberta-v3-base",
        "--strict-projector",
    ]
    if require_semantic:
        argv.append("--require-semantic")
    if require_calibration:
        argv.append("--require-pitheta-calibration")

    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    payload = _extract_json_blob(proc.stdout)
    payload["_exit_code"] = int(proc.returncode)
    payload["_stdout"] = proc.stdout
    payload["_stderr"] = proc.stderr
    return payload


def _gate_to_dict(gate: GateResult) -> Dict[str, Any]:
    return {
        "gate_id": gate.gate_id,
        "status": gate.status,
        "observed": gate.observed,
        "threshold": gate.threshold,
        "message": gate.message,
    }


def _ece_binary(probs: np.ndarray, labels: np.ndarray, *, bins: int = 10) -> float:
    if probs.size == 0 or labels.size == 0:
        return 0.0
    p = np.clip(probs.astype(np.float64), 0.0, 1.0)
    y = np.clip(labels.astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, int(max(2, bins)) + 1)
    ece = 0.0
    total = float(len(p))
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if i < len(edges) - 2:
            mask = (p >= lo) & (p < hi)
        else:
            mask = (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count <= 0:
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        ece += abs(acc - conf) * (float(count) / total)
    return float(ece)


def _resolve_base_model(checkpoint_dir: str) -> str:
    manifest_path = Path(checkpoint_dir) / "model_manifest.json"
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            base = str(payload.get("base_model", "")).strip()
            if base:
                return base
        except Exception:
            pass
    return "deberta-v3-base"


def _parse_nll(calibration_payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key in ("ordinal", "polarity"):
        block = calibration_payload.get(key, {}) if isinstance(calibration_payload, dict) else {}
        if not isinstance(block, dict):
            continue
        out[key] = {
            "nll_before": float(block.get("nll_before", 0.0)),
            "nll_after": float(block.get("nll_after", 0.0)),
            "temperatures": [float(x) for x in list(block.get("temperatures", []))],
        }
    return out


def _compute_pitheta_calibration_report(
    *,
    checkpoint_dir: str,
    data_dir: str,
    pressure_map: List[float],
) -> Dict[str, Any]:
    holdout_path = Path(data_dir) / "holdout.jsonl"
    if not holdout_path.exists():
        return {"status": "unavailable", "reason": "holdout.jsonl_not_found"}
    rows = load_pitheta_jsonl(holdout_path.as_posix())
    if not rows:
        return {"status": "unavailable", "reason": "holdout.jsonl_empty"}

    runtime = PiThetaInference(
        PiThetaInferenceConfig(
            checkpoint_dir=str(checkpoint_dir),
            base_model_path=_resolve_base_model(checkpoint_dir),
            max_length=256,
            batch_size=16,
            device="auto",
            local_files_only=True,
            require_calibration=False,
        )
    )
    if not runtime.ready:
        return {"status": "unavailable", "reason": str(runtime.runtime_error or "pitheta_runtime_not_ready")}

    texts = [str(r.get("text", "")) for r in rows]
    labels = np.asarray([list(map(int, r.get("wall_labels", [0, 0, 0, 0]))) for r in rows], dtype=np.int64)
    outputs = runtime.predict_outputs(texts)
    polarity_prob = np.asarray(outputs.get("polarity_prob", np.zeros((len(rows), 4, 3), dtype=np.float32)), dtype=np.float32)
    p_directive = np.clip(polarity_prob[:, :, 2], 0.0, 1.0)
    head_version = str(outputs.get("head_version", "legacy_v1"))
    if head_version == "ordinal_v2":
        ord_prob = np.asarray(outputs.get("ordinal_prob", np.zeros((len(rows), 4, 4), dtype=np.float32)), dtype=np.float32)
        p_map = np.asarray([float(x) for x in pressure_map], dtype=np.float32).reshape((1, 1, 4))
        expected_pressure = np.sum(ord_prob * p_map, axis=2)
        p_wall = np.clip(expected_pressure * p_directive, 0.0, 1.0)
    else:
        wall_prob = np.asarray(outputs.get("wall_prob", np.zeros((len(rows), 4), dtype=np.float32)), dtype=np.float32)
        p_wall = np.clip(wall_prob * p_directive, 0.0, 1.0)

    walls = [
        "override_instructions",
        "secret_exfiltration",
        "tool_or_action_abuse",
        "policy_evasion",
    ]
    ece_per_wall = {
        walls[i]: _ece_binary(p_wall[:, i], labels[:, i], bins=10)
        for i in range(4)
    }

    calibration_payload = runtime.calibration_payload
    return {
        "status": "ok",
        "samples": int(len(rows)),
        "head_version": head_version,
        "calibrated": bool(runtime.calibration_active),
        "nll": _parse_nll(calibration_payload),
        "ece_per_wall": ece_per_wall,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate PiTheta checkpoint against baseline gates.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--deepset-root", default="data/deepset-prompt-injections")
    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-semantic", action="store_true")
    parser.add_argument("--require-calibration", action="store_true")
    parser.add_argument("--require-omega-compat", action="store_true")
    args = parser.parse_args()
    require_calibration = bool(args.require_calibration or args.strict_gates)
    require_omega_compat = bool(args.require_omega_compat or args.strict_gates)

    baseline = json.loads(Path(args.baseline_report).read_text(encoding="utf-8"))
    candidate = _run_eval(
        profile=str(args.profile),
        checkpoint_dir=str(args.checkpoint),
        deepset_root=str(args.deepset_root),
        require_semantic=bool(args.require_semantic),
        require_calibration=require_calibration,
    )
    status, gates = evaluate_pitheta_gates(candidate, baseline)
    gate_ids_enabled = {"PTG-001", "PTG-002", "PTG-003", "PTG-004", "PTG-005", "PTG-006"}
    if require_calibration:
        gate_ids_enabled.add("PTG-007")
    if require_omega_compat:
        gate_ids_enabled.update({"PTG-008", "PTG-009"})
    filtered_gates = []
    for gate in gates:
        if gate.gate_id in gate_ids_enabled:
            filtered_gates.append(gate)
        else:
            filtered_gates.append(
                GateResult(
                    gate_id=gate.gate_id,
                    status="PASS",
                    observed=gate.observed,
                    threshold=gate.threshold,
                    message=f"{gate.message} (skipped)",
                )
            )
    gates = filtered_gates
    status = "GO" if all(g.status == "PASS" for g in gates) else "NO_GO"
    pressure_map = [0.0, 0.25, 0.6, 1.0]
    conv = candidate.get("pitheta_conversion", {}) if isinstance(candidate, dict) else {}
    if isinstance(conv, dict) and isinstance(conv.get("pressure_map"), list) and len(conv.get("pressure_map", [])) == 4:
        pressure_map = [float(x) for x in conv.get("pressure_map", [0.0, 0.25, 0.6, 1.0])]
    pitheta_calibration = _compute_pitheta_calibration_report(
        checkpoint_dir=str(args.checkpoint),
        data_dir=str(args.data_dir),
        pressure_map=pressure_map,
    )

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir or (ROOT / "artifacts" / "pitheta_eval" / f"eval_{run_id}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_report = {
        "run_id": run_id,
        "status": status,
        "profile": args.profile,
        "checkpoint": str(Path(args.checkpoint).as_posix()),
        "data_dir": str(Path(args.data_dir).as_posix()),
        "baseline_report": str(Path(args.baseline_report).as_posix()),
        "candidate_metrics": {
            "hard_negatives": candidate.get("hard_negatives", {}),
            "whitebox": candidate.get("whitebox", {}),
            "deepset": candidate.get("deepset", {}),
        },
        "baseline_metrics": {
            "hard_negatives": baseline.get("hard_negatives", {}),
            "whitebox": baseline.get("whitebox", {}),
            "deepset": baseline.get("deepset", {}),
        },
        "gates": [_gate_to_dict(g) for g in gates],
        "run_eval_exit_code": int(candidate.get("_exit_code", 1)),
        "require_calibration": require_calibration,
        "require_omega_compat": require_omega_compat,
        "pitheta_calibration": pitheta_calibration,
    }

    gate_report = {
        "status": status,
        "strict": bool(args.strict_gates),
        "failed_gates": [g.gate_id for g in gates if g.status != "PASS"],
    }
    (out_dir / "eval_report.json").write_text(json.dumps(eval_report, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "gate_report.json").write_text(json.dumps(gate_report, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "output_dir": out_dir.as_posix(), "failed_gates": gate_report["failed_gates"]}, ensure_ascii=True, indent=2))

    if args.strict_gates and status != "GO":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
