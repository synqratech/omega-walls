from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import config_refs_from_snapshot, load_resolved_config


@dataclass
class CommandRun:
    name: str
    argv: List[str]
    exit_code: int
    duration_sec: float
    stdout_file: str
    stderr_file: str


def _pick_python() -> str:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        try:
            probe = subprocess.run(
                [str(venv_python), "--version"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if int(probe.returncode) == 0:
                return str(venv_python)
        except Exception:
            pass
    return sys.executable


def _run_and_capture(name: str, argv: List[str], out_dir: Path) -> CommandRun:
    started = time.time()
    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    duration = round(time.time() - started, 3)
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return CommandRun(
        name=name,
        argv=list(argv),
        exit_code=int(proc.returncode),
        duration_sec=duration,
        stdout_file=str(stdout_path.relative_to(out_dir).as_posix()),
        stderr_file=str(stderr_path.relative_to(out_dir).as_posix()),
    )


def _extract_json_payload(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("stdout does not contain JSON object payload")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("parsed payload is not an object")
    return payload


def _as_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT).as_posix())
    except Exception:
        return str(path.as_posix())


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _latest_release_metrics_path() -> Path | None:
    root = ROOT / "artifacts" / "release_gate"
    if not root.exists():
        return None
    candidates = sorted(
        [p / "release_metrics.json" for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _threshold_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    omega_off = ((cfg.get("omega", {}) or {}).get("off", {}) or {})
    deepset_thresholds = (((cfg.get("deepset", {}) or {}).get("thresholds", {}) or {}).get("report", {}) or {})
    release_gate_cfg = (cfg.get("release_gate", {}) or {}).get("gates", [])
    release_core: Dict[str, Dict[str, Any]] = {}
    if isinstance(release_gate_cfg, list):
        for gate in release_gate_cfg:
            if not isinstance(gate, dict):
                continue
            gate_id = str(gate.get("id", "")).strip()
            if not gate_id:
                continue
            release_core[gate_id] = {
                "metric": gate.get("metric"),
                "op": gate.get("op"),
                "threshold": gate.get("threshold"),
            }
    return {
        "projector": {
            "mode": ((cfg.get("projector", {}) or {}).get("mode", "pi0")),
            "fallback_to_pi0": ((cfg.get("projector", {}) or {}).get("fallback_to_pi0", True)),
        },
        "omega_off": {
            "tau": float(omega_off.get("tau", 0.90)),
            "Theta": float(omega_off.get("Theta", 0.80)),
            "Sigma": float(omega_off.get("Sigma", 0.90)),
        },
        "deepset": {
            "split_default": (cfg.get("deepset", {}) or {}).get("split_default", "test"),
            "mode_default": (cfg.get("deepset", {}) or {}).get("mode_default", "full"),
            "seed_default": (((cfg.get("deepset", {}) or {}).get("reproducibility", {}) or {}).get("seed_default", 41)),
            "thresholds_report": dict(deepset_thresholds),
        },
        "release_gate_gates": release_core,
    }


def _metric_summary(
    run_eval_report: Dict[str, Any],
    deepset_report: Dict[str, Any],
    fn_report: Dict[str, Any],
    pareto_report: Dict[str, Any],
) -> Dict[str, Any]:
    deepset_metrics = (deepset_report.get("metrics", {}) or {}) if isinstance(deepset_report, dict) else {}
    whitebox = (run_eval_report.get("whitebox", {}) or {}) if isinstance(run_eval_report, dict) else {}
    hard_neg = (run_eval_report.get("hard_negatives", {}) or {}) if isinstance(run_eval_report, dict) else {}
    fn_summary = (fn_report.get("summary", {}) or {}) if isinstance(fn_report, dict) else {}
    pareto_summary = (pareto_report.get("summary", {}) or {}) if isinstance(pareto_report, dict) else {}
    confusion = (deepset_report.get("confusion_matrix", {}) or {}) if isinstance(deepset_report, dict) else {}
    return {
        "deepset": {
            "attack_off_rate": deepset_metrics.get("attack_off_rate"),
            "benign_off_rate": deepset_metrics.get("benign_off_rate"),
            "f1": deepset_metrics.get("f1"),
            "coverage_wall_any_attack": deepset_metrics.get("coverage_wall_any_attack"),
            "fn_total": confusion.get("fn"),
            "tp": confusion.get("tp"),
            "fp": confusion.get("fp"),
            "tn": confusion.get("tn"),
        },
        "whitebox": {
            "evaluated": whitebox.get("evaluated"),
            "base_detect_rate": whitebox.get("base_detect_rate"),
            "bypass_rate": whitebox.get("bypass_rate"),
        },
        "hard_negatives": {
            "fp": hard_neg.get("fp"),
            "count": hard_neg.get("count"),
        },
        "fn_analysis": {
            "fn_total": fn_summary.get("fn_total"),
            "fn_rate": fn_summary.get("fn_rate"),
        },
        "pareto": {
            "selected_rule_packs": pareto_summary.get("selected_rule_packs"),
            "achieved_fn_coverage": pareto_summary.get("achieved_fn_coverage"),
            "actionable_fn_coverage_no_generic": pareto_summary.get("actionable_fn_coverage_no_generic"),
        },
    }


def _compare_summary(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    tolerance: float,
) -> Tuple[bool, List[Dict[str, Any]]]:
    tracked = [
        ("deepset.attack_off_rate", baseline.get("deepset", {}).get("attack_off_rate"), current.get("deepset", {}).get("attack_off_rate")),
        ("deepset.benign_off_rate", baseline.get("deepset", {}).get("benign_off_rate"), current.get("deepset", {}).get("benign_off_rate")),
        ("deepset.fn_total", baseline.get("deepset", {}).get("fn_total"), current.get("deepset", {}).get("fn_total")),
        ("whitebox.base_detect_rate", baseline.get("whitebox", {}).get("base_detect_rate"), current.get("whitebox", {}).get("base_detect_rate")),
        ("whitebox.bypass_rate", baseline.get("whitebox", {}).get("bypass_rate"), current.get("whitebox", {}).get("bypass_rate")),
        ("hard_negatives.fp", baseline.get("hard_negatives", {}).get("fp"), current.get("hard_negatives", {}).get("fp")),
    ]
    diffs: List[Dict[str, Any]] = []
    ok = True
    for name, bval, cval in tracked:
        if isinstance(bval, (int, float)) and isinstance(cval, (int, float)):
            delta = float(cval) - float(bval)
            match = abs(delta) <= tolerance
            if not match:
                ok = False
            diffs.append(
                {
                    "metric": name,
                    "baseline": float(bval),
                    "current": float(cval),
                    "delta": delta,
                    "within_tolerance": match,
                }
            )
        else:
            match = bval == cval
            if not match:
                ok = False
            diffs.append(
                {
                    "metric": name,
                    "baseline": bval,
                    "current": cval,
                    "delta": None,
                    "within_tolerance": match,
                }
            )
    return ok, diffs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reproducible rule-hardening cycle: run_eval -> analyze_deepset_fn -> extract_rule_pareto"
    )
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--label", default="rule_cycle")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--projector-mode", choices=["pi0", "pitheta", "hybrid"], default="pi0")
    parser.add_argument("--semantic-model-path", default="e5-small-v2")
    parser.add_argument("--deepset-benchmark-root", default="data/deepset-prompt-injections")
    parser.add_argument("--deepset-split", default="test")
    parser.add_argument("--deepset-mode", choices=["sampled", "full"], default="full")
    parser.add_argument("--deepset-max-samples", type=int, default=116)
    parser.add_argument("--whitebox-max-samples", type=int, default=200)
    parser.add_argument("--whitebox-max-iters", type=int, default=5)
    parser.add_argument("--whitebox-beam-width", type=int, default=4)
    parser.add_argument("--whitebox-mutations", type=int, default=3)
    parser.add_argument("--target-fn-coverage", type=float, default=0.80)
    parser.add_argument("--release-metrics-json", default=None)
    parser.add_argument("--artifacts-root", default="artifacts/rule_cycle")
    parser.add_argument("--freeze-baseline", action="store_true")
    parser.add_argument("--baseline-pointer", default="artifacts/rule_cycle/BASELINE_LATEST.json")
    parser.add_argument("--baseline-manifest", default=None)
    parser.add_argument("--require-reproducible", action="store_true")
    parser.add_argument("--repro-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    def _emit_stage_failure(stage: str, message: str, *, details: Dict[str, Any] | None = None) -> int:
        payload: Dict[str, Any] = {
            "run_id": run_id,
            "status": "FAILED",
            "failed_stage": stage,
            "error": message,
        }
        if isinstance(details, dict) and details:
            payload["details"] = details
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    cli_overrides = {
        "projector": {"mode": str(args.projector_mode)},
        "pi0": {"semantic": {"model_path": str(args.semantic_model_path)}},
    }
    snapshot = load_resolved_config(profile=str(args.profile), cli_overrides=cli_overrides)
    utc_now = datetime.now(timezone.utc)
    run_id = f"{args.label}_{utc_now.strftime('%Y%m%dT%H%M%SZ')}_{snapshot.resolved_sha256[:12]}"
    run_dir = ROOT / str(args.artifacts_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    python_exec = _pick_python()
    deepset_json = run_dir / "deepset_report.json"

    run_eval_cmd = [
        python_exec,
        "scripts/run_eval.py",
        "--profile",
        str(args.profile),
        "--projector-mode",
        str(args.projector_mode),
        "--run-deepset",
        "--deepset-benchmark-root",
        str(args.deepset_benchmark_root),
        "--deepset-split",
        str(args.deepset_split),
        "--deepset-mode",
        str(args.deepset_mode),
        "--deepset-max-samples",
        str(int(args.deepset_max_samples)),
        "--deepset-seed",
        str(int(args.seed)),
        "--require-semantic",
        "--semantic-model-path",
        str(args.semantic_model_path),
        "--enforce-whitebox",
        "--whitebox-max-samples",
        str(int(args.whitebox_max_samples)),
        "--whitebox-max-iters",
        str(int(args.whitebox_max_iters)),
        "--whitebox-beam-width",
        str(int(args.whitebox_beam_width)),
        "--whitebox-mutations",
        str(int(args.whitebox_mutations)),
        "--deepset-json-output",
        str(deepset_json.as_posix()),
    ]
    cmd_runs: List[CommandRun] = []

    run_eval_res = _run_and_capture("run_eval", run_eval_cmd, run_dir)
    cmd_runs.append(run_eval_res)
    if int(run_eval_res.exit_code) != 0:
        return _emit_stage_failure(
            "run_eval",
            "run_eval command failed",
            details={
                "exit_code": int(run_eval_res.exit_code),
                "stdout_file": str(run_eval_res.stdout_file),
                "stderr_file": str(run_eval_res.stderr_file),
            },
        )
    run_eval_stdout = (run_dir / run_eval_res.stdout_file).read_text(encoding="utf-8", errors="replace")
    try:
        run_eval_report = _extract_json_payload(run_eval_stdout)
    except Exception as exc:
        return _emit_stage_failure(
            "run_eval",
            "run_eval stdout does not contain JSON payload",
            details={
                "error": str(exc),
                "stdout_file": str(run_eval_res.stdout_file),
                "stderr_file": str(run_eval_res.stderr_file),
            },
        )
    (run_dir / "run_eval_report.json").write_text(
        json.dumps(run_eval_report, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    deepset_report = json.loads(deepset_json.read_text(encoding="utf-8")) if deepset_json.exists() else {}

    analyze_cmd = [
        python_exec,
        "scripts/analyze_deepset_fn.py",
        "--profile",
        str(args.profile),
        "--benchmark-root",
        str(args.deepset_benchmark_root),
        "--split",
        str(args.deepset_split),
        "--mode",
        str(args.deepset_mode),
        "--seed",
        str(int(args.seed)),
        "--max-samples",
        str(int(args.deepset_max_samples)),
        "--require-semantic",
    ]
    analyze_res = _run_and_capture("analyze_deepset_fn", analyze_cmd, run_dir)
    cmd_runs.append(analyze_res)
    if int(analyze_res.exit_code) != 0:
        return _emit_stage_failure(
            "analyze_deepset_fn",
            "analyze_deepset_fn command failed",
            details={
                "exit_code": int(analyze_res.exit_code),
                "stdout_file": str(analyze_res.stdout_file),
                "stderr_file": str(analyze_res.stderr_file),
            },
        )
    analyze_stdout = (run_dir / analyze_res.stdout_file).read_text(encoding="utf-8", errors="replace")
    try:
        fn_report = _extract_json_payload(analyze_stdout)
    except Exception as exc:
        return _emit_stage_failure(
            "analyze_deepset_fn",
            "analyze_deepset_fn stdout does not contain JSON payload",
            details={
                "error": str(exc),
                "stdout_file": str(analyze_res.stdout_file),
                "stderr_file": str(analyze_res.stderr_file),
            },
        )
    (run_dir / "fn_analysis_report.json").write_text(
        json.dumps(fn_report, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    fn_artifacts = fn_report.get("artifacts", {}) if isinstance(fn_report, dict) else {}
    fn_status = str((fn_report or {}).get("status", "")).strip().upper()
    if fn_status and fn_status not in {"OK", "GO"}:
        return _emit_stage_failure(
            "analyze_deepset_fn",
            "FN analysis returned non-OK status",
            details={"status": fn_status, "report": fn_report},
        )
    fn_samples = Path(str(fn_artifacts.get("fn_samples_jsonl", "")))
    fn_patterns = Path(str(fn_artifacts.get("fn_patterns_json", "")))
    if (not str(fn_samples)) or (not str(fn_patterns)):
        return _emit_stage_failure(
            "analyze_deepset_fn",
            "FN analysis did not provide required artifacts",
            details={"artifacts": fn_artifacts},
        )
    if (not fn_samples.exists()) or (not fn_patterns.exists()):
        return _emit_stage_failure(
            "analyze_deepset_fn",
            "FN artifact paths are missing",
            details={
                "fn_samples_jsonl": str(fn_samples.as_posix()),
                "fn_patterns_json": str(fn_patterns.as_posix()),
            },
        )
    release_metrics_path = Path(str(args.release_metrics_json)) if args.release_metrics_json else _latest_release_metrics_path()
    if release_metrics_path is None or not release_metrics_path.exists():
        return _emit_stage_failure(
            "extract_rule_pareto",
            "release_metrics.json is required for pareto step (pass --release-metrics-json)",
        )

    pareto_cmd = [
        python_exec,
        "scripts/extract_rule_pareto.py",
        "--fn-samples-jsonl",
        str(fn_samples.as_posix()),
        "--fn-patterns-json",
        str(fn_patterns.as_posix()),
        "--release-metrics-json",
        str(release_metrics_path.as_posix()),
        "--target-fn-coverage",
        str(float(args.target_fn_coverage)),
    ]
    pareto_res = _run_and_capture("extract_rule_pareto", pareto_cmd, run_dir)
    cmd_runs.append(pareto_res)
    if int(pareto_res.exit_code) != 0:
        return _emit_stage_failure(
            "extract_rule_pareto",
            "extract_rule_pareto command failed",
            details={
                "exit_code": int(pareto_res.exit_code),
                "stdout_file": str(pareto_res.stdout_file),
                "stderr_file": str(pareto_res.stderr_file),
            },
        )
    pareto_stdout = (run_dir / pareto_res.stdout_file).read_text(encoding="utf-8", errors="replace")
    try:
        pareto_report = _extract_json_payload(pareto_stdout)
    except Exception as exc:
        return _emit_stage_failure(
            "extract_rule_pareto",
            "extract_rule_pareto stdout does not contain JSON payload",
            details={
                "error": str(exc),
                "stdout_file": str(pareto_res.stdout_file),
                "stderr_file": str(pareto_res.stderr_file),
            },
        )
    (run_dir / "rule_pareto_report.json").write_text(
        json.dumps(pareto_report, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    pareto_run_id = str(pareto_report.get("run_id", "")).strip()
    pareto_report_path = (
        ROOT / "artifacts" / "rule_pareto" / pareto_run_id / "rule_pareto_report.json" if pareto_run_id else None
    )
    pareto_md_path = (
        ROOT / "artifacts" / "rule_pareto" / pareto_run_id / "rule_pareto_report.md" if pareto_run_id else None
    )

    baseline_dir = run_dir / "baseline_snapshot"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    _copy_if_exists(run_dir / "run_eval_report.json", baseline_dir / "run_eval_report.json")
    _copy_if_exists(deepset_json, baseline_dir / "deepset_report.json")
    _copy_if_exists(fn_samples, baseline_dir / "fn_samples.jsonl")
    _copy_if_exists(fn_patterns, baseline_dir / "fn_patterns.json")
    _copy_if_exists(fn_patterns.with_suffix(".md"), baseline_dir / "fn_patterns.md")
    if pareto_report_path is not None:
        _copy_if_exists(pareto_report_path, baseline_dir / "rule_pareto_report.json")
    if pareto_md_path is not None:
        _copy_if_exists(pareto_md_path, baseline_dir / "rule_pareto_report.md")

    threshold_snapshot = _threshold_snapshot(snapshot.resolved)
    threshold_path = baseline_dir / "threshold_snapshot.json"
    threshold_path.write_text(json.dumps(threshold_snapshot, ensure_ascii=True, indent=2), encoding="utf-8")
    resolved_path = baseline_dir / "resolved_config.json"
    resolved_path.write_text(
        json.dumps(snapshot.resolved, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )

    summary = _metric_summary(
        run_eval_report=run_eval_report,
        deepset_report=deepset_report,
        fn_report=fn_report,
        pareto_report=pareto_report,
    )

    repro_result: Dict[str, Any] = {
        "checked": False,
        "baseline_manifest": None,
        "tolerance": float(args.repro_tolerance),
        "pass": None,
        "diffs": [],
    }
    if args.baseline_manifest:
        baseline_manifest = Path(str(args.baseline_manifest))
        if not baseline_manifest.is_absolute():
            baseline_manifest = ROOT / baseline_manifest
        baseline_payload = json.loads(baseline_manifest.read_text(encoding="utf-8"))
        baseline_summary = baseline_payload.get("summary_metrics", {})
        repro_ok, repro_diffs = _compare_summary(
            baseline=baseline_summary if isinstance(baseline_summary, dict) else {},
            current=summary,
            tolerance=float(args.repro_tolerance),
        )
        repro_result = {
            "checked": True,
            "baseline_manifest": _as_rel(baseline_manifest),
            "tolerance": float(args.repro_tolerance),
            "pass": bool(repro_ok),
            "diffs": repro_diffs,
        }

    commands_failed = any(int(c.exit_code) != 0 for c in cmd_runs)
    reproducibility_failed = bool(args.require_reproducible) and bool(repro_result.get("checked")) and not bool(
        repro_result.get("pass")
    )
    status = "OK" if (not commands_failed and not reproducibility_failed) else "FAILED"

    manifest = {
        "run_id": run_id,
        "created_utc": utc_now.isoformat(),
        "status": status,
        "profile": str(args.profile),
        "seed": int(args.seed),
        "config_refs": config_refs_from_snapshot(snapshot, code_commit="local"),
        "cycle": {
            "projector_mode": str(args.projector_mode),
            "semantic_model_path": str(args.semantic_model_path),
            "deepset_benchmark_root": str(args.deepset_benchmark_root),
            "deepset_split": str(args.deepset_split),
            "deepset_mode": str(args.deepset_mode),
            "deepset_max_samples": int(args.deepset_max_samples),
            "whitebox_max_samples": int(args.whitebox_max_samples),
            "whitebox_max_iters": int(args.whitebox_max_iters),
            "whitebox_beam_width": int(args.whitebox_beam_width),
            "whitebox_mutations": int(args.whitebox_mutations),
            "target_fn_coverage": float(args.target_fn_coverage),
        },
        "artifacts": {
            "run_eval_report": _as_rel(run_dir / "run_eval_report.json"),
            "deepset_report": _as_rel(deepset_json),
            "fn_analysis_report": _as_rel(run_dir / "fn_analysis_report.json"),
            "rule_pareto_report": _as_rel(run_dir / "rule_pareto_report.json"),
            "baseline_snapshot_dir": _as_rel(baseline_dir),
            "threshold_snapshot": _as_rel(threshold_path),
            "resolved_config": _as_rel(resolved_path),
            "source_fn_samples": _as_rel(fn_samples),
            "source_fn_patterns": _as_rel(fn_patterns),
            "source_rule_pareto_report": _as_rel(pareto_report_path) if pareto_report_path else None,
        },
        "commands": [asdict(c) for c in cmd_runs],
        "summary_metrics": summary,
        "reproducibility": repro_result,
    }

    manifest_path = run_dir / "cycle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    latest_path = ROOT / str(args.artifacts_root) / "LATEST.json"
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "manifest": _as_rel(manifest_path),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    if args.freeze_baseline:
        baseline_ptr = Path(str(args.baseline_pointer))
        if not baseline_ptr.is_absolute():
            baseline_ptr = ROOT / baseline_ptr
        baseline_ptr.parent.mkdir(parents=True, exist_ok=True)
        baseline_ptr.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "status": status,
                    "manifest": _as_rel(manifest_path),
                    "summary_metrics": summary,
                    "seed": int(args.seed),
                    "profile": str(args.profile),
                },
                ensure_ascii=True,
                indent=2,
            ),
            encoding="utf-8",
        )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "manifest": _as_rel(manifest_path),
                "latest": _as_rel(latest_path),
                "baseline_pointer": str(args.baseline_pointer) if args.freeze_baseline else None,
            },
            ensure_ascii=True,
            indent=2,
        )
    )

    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
