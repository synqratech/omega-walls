from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.pitheta.gold_slice import load_gold_slice_jsonl


def _extract_json_blob(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("unable to locate JSON in command output")
    return json.loads(text[start : end + 1])


def _run_cmd(argv: List[str], *, timeout_sec: int = 7200) -> Tuple[int, str, str]:
    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=int(timeout_sec),
    )
    return int(proc.returncode), str(proc.stdout), str(proc.stderr)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_nll_improvement(cal_report: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
    nll = cal_report.get("nll", {}) if isinstance(cal_report, dict) else {}
    out: Dict[str, bool] = {}
    ok = True
    for key in ("ordinal", "polarity"):
        block = nll.get(key, {}) if isinstance(nll, dict) else {}
        before = float(block.get("nll_before", 0.0))
        after = float(block.get("nll_after", 0.0))
        good = after < before
        out[key] = bool(good)
        if not good:
            ok = False
    return ok, out


def _check_ece_non_worse(calibrated: Dict[str, Any], uncalibrated: Dict[str, Any]) -> Tuple[bool, Dict[str, bool]]:
    c = (calibrated.get("ece_per_wall", {}) if isinstance(calibrated, dict) else {}) or {}
    u = (uncalibrated.get("ece_per_wall", {}) if isinstance(uncalibrated, dict) else {}) or {}
    walls = sorted(set(c.keys()).intersection(set(u.keys())))
    if not walls:
        return False, {"_missing": False}
    out: Dict[str, bool] = {}
    ok = True
    for wall in walls:
        good = float(c.get(wall, 1.0)) <= float(u.get(wall, 0.0)) + 1e-9
        out[wall] = bool(good)
        if not good:
            ok = False
    return ok, out


def _kappa_gate(
    *,
    report_path: Path,
    ord_thr: float,
    pol_thr: float,
) -> Tuple[bool, Dict[str, Any]]:
    payload = _read_json(report_path)
    agreement = payload.get("agreement", {}) if isinstance(payload, dict) else {}
    ord_map = agreement.get("ordinal_quadratic_kappa_per_wall", {}) or {}
    pol_map = agreement.get("polarity_quadratic_kappa_per_wall", {}) or {}
    verdict = True
    failed: List[str] = []
    for wall, value in ord_map.items():
        if float(value) < float(ord_thr):
            verdict = False
            failed.append(f"ordinal:{wall}")
    for wall, value in pol_map.items():
        if float(value) < float(pol_thr):
            verdict = False
            failed.append(f"polarity:{wall}")
    independence = payload.get("independence", {}) if isinstance(payload, dict) else {}
    if bool(independence.get("identical_annotations", False)):
        verdict = False
        failed.append("independence:annotator_a_and_b_identical")
    return verdict, {"ordinal": ord_map, "polarity": pol_map, "failed": failed}


def _md(report: Dict[str, Any]) -> str:
    lines = [
        "# PiTheta Precloud Gate",
        "",
        f"- status: `{report.get('status', 'NO_GO')}`",
        f"- run_id: `{report.get('run_id', '')}`",
        "",
        "## Checks",
    ]
    for name, value in (report.get("checks", {}) or {}).items():
        lines.append(f"- `{name}`: `{value}`")
    lines.append("")
    lines.append("## Artifacts")
    for name, value in (report.get("artifacts", {}) or {}).items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-cloud GO/NO-GO gate for PiTheta.")
    parser.add_argument("--gold-slice-path", default="data/gold_slice/gold_slice.jsonl")
    parser.add_argument("--min-gold-size", type=int, default=200)
    parser.add_argument("--kappa-report", default=None)
    parser.add_argument("--kappa-ord-threshold", type=float, default=0.70)
    parser.add_argument("--kappa-pol-threshold", type=float, default=0.65)
    parser.add_argument("--build-dataset", action="store_true")
    parser.add_argument("--dataset-output-dir", default="artifacts/pitheta_data/run_gold_blended")
    parser.add_argument("--build-calibration-split", action="store_true")
    parser.add_argument("--calibration-source-splits", default="train,dev")
    parser.add_argument("--calibration-target-size", type=int, default=600)
    parser.add_argument("--calibration-seed", type=int, default=41)
    parser.add_argument("--train-smoke", action="store_true")
    parser.add_argument("--train-output-dir", default="artifacts/pitheta_train/run_gold_blended_smoke")
    parser.add_argument("--train-config", default="config/pitheta_train.yml")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--train-temperature-split", choices=["dev", "holdout", "calibration"], default="dev")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--baseline-report", default="artifacts/deepset_eval/latest_deepset_report.json")
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--strict-gates", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir or (ROOT / "artifacts" / "precloud_gate" / f"run_{run_id}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: Dict[str, Any] = {}
    artifacts: Dict[str, Any] = {}
    failures: List[str] = []

    gold_rows = load_gold_slice_jsonl(str(args.gold_slice_path))
    checks["gold_slice_size"] = int(len(gold_rows))
    checks["gold_slice_min_required"] = int(args.min_gold_size)
    if len(gold_rows) < int(args.min_gold_size):
        failures.append(f"gold_slice_size<{args.min_gold_size}")

    if args.kappa_report:
        kappa_ok, kappa_payload = _kappa_gate(
            report_path=Path(args.kappa_report),
            ord_thr=float(args.kappa_ord_threshold),
            pol_thr=float(args.kappa_pol_threshold),
        )
        checks["kappa_gate_pass"] = bool(kappa_ok)
        checks["kappa_failed"] = list(kappa_payload.get("failed", []))
        artifacts["kappa_report"] = str(Path(args.kappa_report).as_posix())
        if not kappa_ok:
            failures.append("kappa_gate_failed")
    else:
        checks["kappa_gate_pass"] = "skipped"

    if args.build_dataset:
        rc, stdout, stderr = _run_cmd(
            [
                sys.executable,
                "scripts/build_pitheta_dataset.py",
                "--registry",
                "config/pitheta_dataset_registry.yml",
                "--output-dir",
                str(args.dataset_output_dir),
                "--seed",
                "41",
                "--strict",
            ],
            timeout_sec=3600,
        )
        (out_dir / "build_dataset.stdout.txt").write_text(stdout, encoding="utf-8")
        (out_dir / "build_dataset.stderr.txt").write_text(stderr, encoding="utf-8")
        checks["build_dataset_rc"] = int(rc)
        artifacts["build_dataset_stdout"] = (out_dir / "build_dataset.stdout.txt").as_posix()
        if rc != 0:
            failures.append("build_dataset_failed")

    if args.build_calibration_split:
        rc, stdout, stderr = _run_cmd(
            [
                sys.executable,
                "scripts/build_pitheta_calibration_split.py",
                "--data-dir",
                str(args.dataset_output_dir),
                "--source-splits",
                str(args.calibration_source_splits),
                "--target-size",
                str(int(args.calibration_target_size)),
                "--seed",
                str(int(args.calibration_seed)),
                "--strict",
            ],
            timeout_sec=3600,
        )
        (out_dir / "build_calibration.stdout.txt").write_text(stdout, encoding="utf-8")
        (out_dir / "build_calibration.stderr.txt").write_text(stderr, encoding="utf-8")
        checks["build_calibration_rc"] = int(rc)
        artifacts["build_calibration_stdout"] = (out_dir / "build_calibration.stdout.txt").as_posix()
        if rc != 0:
            failures.append("build_calibration_failed")

    if args.train_smoke:
        rc, stdout, stderr = _run_cmd(
            [
                sys.executable,
                "scripts/train_pitheta_lora.py",
                "--train-config",
                str(args.train_config),
                "--data-dir",
                str(args.dataset_output_dir),
                "--output-dir",
                str(args.train_output_dir),
                "--epochs",
                str(int(args.epochs)),
                "--fit-temperature",
                "true",
                "--temperature-split",
                str(args.train_temperature_split),
                "--calibration-source-mode",
                "blended",
                "--calibration-gold-slice-path",
                str(args.gold_slice_path),
                "--calibration-gold-ratio",
                "0.70",
                "--calibration-weak-ratio",
                "0.30",
            ],
            timeout_sec=7200,
        )
        (out_dir / "train_smoke.stdout.txt").write_text(stdout, encoding="utf-8")
        (out_dir / "train_smoke.stderr.txt").write_text(stderr, encoding="utf-8")
        checks["train_smoke_rc"] = int(rc)
        artifacts["train_smoke_stdout"] = (out_dir / "train_smoke.stdout.txt").as_posix()
        if rc != 0:
            failures.append("train_smoke_failed")

    checkpoint = str(args.checkpoint or (Path(args.train_output_dir) / "best").as_posix())
    eval_out = Path(args.eval_output_dir or (out_dir / "eval_calibrated"))
    eval_out.mkdir(parents=True, exist_ok=True)
    rc, stdout, stderr = _run_cmd(
        [
            sys.executable,
            "scripts/eval_pitheta.py",
            "--checkpoint",
            checkpoint,
            "--data-dir",
            str(args.dataset_output_dir),
            "--baseline-report",
            str(args.baseline_report),
            "--output-dir",
            eval_out.as_posix(),
            "--require-calibration",
            "--require-omega-compat",
            *(["--strict-gates"] if args.strict_gates else []),
        ],
        timeout_sec=7200,
    )
    (out_dir / "eval_calibrated.stdout.txt").write_text(stdout, encoding="utf-8")
    (out_dir / "eval_calibrated.stderr.txt").write_text(stderr, encoding="utf-8")
    checks["eval_calibrated_rc"] = int(rc)
    artifacts["eval_calibrated_stdout"] = (out_dir / "eval_calibrated.stdout.txt").as_posix()
    if rc != 0:
        failures.append("eval_calibrated_failed")

    calibrated_meta = _extract_json_blob(stdout)
    calibrated_eval = _read_json(Path(calibrated_meta["output_dir"]) / "eval_report.json")
    calibrated_cal = calibrated_eval.get("pitheta_calibration", {})
    nll_ok, nll_map = _check_nll_improvement(calibrated_cal)
    checks["nll_improvement"] = nll_map
    if not nll_ok:
        failures.append("nll_not_improved")

    nocal_ckpt = out_dir / "nocal_ckpt"
    if nocal_ckpt.exists():
        shutil.rmtree(nocal_ckpt)
    shutil.copytree(Path(checkpoint), nocal_ckpt)
    temp_path = nocal_ckpt / "temperature_scaling.json"
    if temp_path.exists():
        temp_path.unlink()
    eval_uncal_out = out_dir / "eval_uncalibrated"
    eval_uncal_out.mkdir(parents=True, exist_ok=True)
    rc_u, stdout_u, stderr_u = _run_cmd(
        [
            sys.executable,
            "scripts/eval_pitheta.py",
            "--checkpoint",
            nocal_ckpt.as_posix(),
            "--data-dir",
            str(args.dataset_output_dir),
            "--baseline-report",
            str(args.baseline_report),
            "--output-dir",
            eval_uncal_out.as_posix(),
            "--require-omega-compat",
        ],
        timeout_sec=7200,
    )
    (out_dir / "eval_uncalibrated.stdout.txt").write_text(stdout_u, encoding="utf-8")
    (out_dir / "eval_uncalibrated.stderr.txt").write_text(stderr_u, encoding="utf-8")
    checks["eval_uncalibrated_rc"] = int(rc_u)
    artifacts["eval_uncalibrated_stdout"] = (out_dir / "eval_uncalibrated.stdout.txt").as_posix()
    if rc_u != 0:
        failures.append("eval_uncalibrated_failed")

    uncal_meta = _extract_json_blob(stdout_u)
    uncal_eval = _read_json(Path(uncal_meta["output_dir"]) / "eval_report.json")
    uncal_cal = uncal_eval.get("pitheta_calibration", {})
    ece_ok, ece_map = _check_ece_non_worse(calibrated_cal, uncal_cal)
    checks["ece_non_worse"] = ece_map
    if not ece_ok:
        failures.append("ece_regression")

    strict_status = str(calibrated_eval.get("status", "NO_GO")).upper()
    checks["strict_eval_status"] = strict_status
    if strict_status != "GO":
        failures.append("strict_eval_not_go")

    report = {
        "run_id": run_id,
        "status": "GO" if not failures else "NO_GO",
        "checks": checks,
        "failures": failures,
        "artifacts": artifacts,
    }
    (out_dir / "go_nogo_report.json").write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    (out_dir / "go_nogo_report.md").write_text(_md(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "output_dir": out_dir.as_posix(), "failures": failures}, ensure_ascii=True, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
