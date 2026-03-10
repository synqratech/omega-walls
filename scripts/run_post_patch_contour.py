from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

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
        raise ValueError("stdout does not contain JSON payload")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def _as_rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT).as_posix())
    except Exception:
        return str(path.as_posix())


def _utc_compact_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _latest_report_json(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    candidates = sorted([p / "report.json" for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _step_record(name: str, cmd: CommandRun, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "name": name,
        "exit_code": int(cmd.exit_code),
        "status": "ok" if int(cmd.exit_code) == 0 else "failed",
        "stdout_file": cmd.stdout_file,
        "stderr_file": cmd.stderr_file,
        "duration_sec": float(cmd.duration_sec),
        "payload": payload,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full post-patch contour and build unified comparative report.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--label", default="post_patch_contour")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--artifacts-root", default="artifacts/post_patch_contour")
    parser.add_argument("--weekly-regression", action="store_true")
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
    parser.add_argument("--bipia-benchmark-root", default="data/BIPIA-main/benchmark")
    parser.add_argument("--bipia-newsqa-data-dir", default=None)
    parser.add_argument("--bipia-mode", choices=["sampled", "full"], default="full")
    parser.add_argument("--bipia-split", default="test")
    parser.add_argument("--bipia-repeats", type=int, default=2)
    parser.add_argument("--pint-dataset", default="data/pint-benchmark/benchmark/data/benchmark_dataset.yaml")
    parser.add_argument("--wainject-root", default="data/WAInjectBench/text")
    parser.add_argument("--wainject-max-samples", type=int, default=0)
    parser.add_argument("--strict-baseline-report", default=None)
    parser.add_argument("--attachment-baseline-report", default=None)
    parser.add_argument("--session-baseline-report", default=None)
    parser.add_argument("--include-session-benchmark", action="store_true")
    parser.add_argument("--session-pack", default="tests/data/session_benchmark/session_pack_seed41_v1.jsonl")
    parser.add_argument("--session-mode", choices=["pi0", "hybrid"], default="pi0")
    parser.add_argument("--release-metrics-json", default=None)
    args = parser.parse_args()

    snapshot = load_resolved_config(
        profile=str(args.profile),
        cli_overrides={"projector": {"mode": "pi0"}, "pi0": {"semantic": {"model_path": str(args.semantic_model_path)}}},
    )
    run_id = f"{args.label}_{_utc_compact_now()}_{snapshot.resolved_sha256[:12]}"
    run_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    python_exec = _pick_python()

    commands: List[CommandRun] = []
    steps: Dict[str, Any] = {}
    reports: Dict[str, Any] = {}

    # 0) BIPIA readiness
    prepare_cmd = [
        python_exec,
        "scripts/prepare_bipia_contexts.py",
        "--benchmark-root",
        str(args.bipia_benchmark_root),
    ]
    if args.bipia_newsqa_data_dir:
        prepare_cmd.extend(["--newsqa-data-dir", str(args.bipia_newsqa_data_dir)])
    prep_run = _run_and_capture("prepare_bipia_contexts", prepare_cmd, run_dir)
    commands.append(prep_run)
    prep_payload: Optional[Dict[str, Any]] = None
    prep_report: Optional[Dict[str, Any]] = None
    try:
        prep_payload = _extract_json_payload((run_dir / prep_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
        prep_report_path = Path(str(prep_payload.get("report", "")))
        if not prep_report_path.is_absolute():
            prep_report_path = (ROOT / prep_report_path).resolve()
        prep_report = _load_json(prep_report_path)
    except Exception as exc:
        prep_payload = {"parse_error": str(exc)}
    steps["prepare_bipia_contexts"] = _step_record("prepare_bipia_contexts", prep_run, prep_payload)
    reports["bipia_prepare"] = prep_report

    bipia_ready = bool((prep_report or {}).get("qa_abstract_md5_ok", False))

    # 1) Rule cycle
    rule_cmd = [
        python_exec,
        "scripts/run_rule_cycle.py",
        "--profile",
        str(args.profile),
        "--label",
        f"{run_id}_rule",
        "--seed",
        str(int(args.seed)),
        "--projector-mode",
        "pi0",
        "--semantic-model-path",
        str(args.semantic_model_path),
        "--deepset-benchmark-root",
        str(args.deepset_benchmark_root),
        "--deepset-split",
        str(args.deepset_split),
        "--deepset-mode",
        str(args.deepset_mode),
        "--deepset-max-samples",
        str(int(args.deepset_max_samples)),
        "--whitebox-max-samples",
        str(int(args.whitebox_max_samples)),
        "--whitebox-max-iters",
        str(int(args.whitebox_max_iters)),
        "--whitebox-beam-width",
        str(int(args.whitebox_beam_width)),
        "--whitebox-mutations",
        str(int(args.whitebox_mutations)),
        "--target-fn-coverage",
        str(float(args.target_fn_coverage)),
    ]
    if args.release_metrics_json:
        rule_cmd.extend(["--release-metrics-json", str(args.release_metrics_json)])
    rule_run = _run_and_capture("run_rule_cycle", rule_cmd, run_dir)
    commands.append(rule_run)
    rule_payload: Optional[Dict[str, Any]] = None
    rule_manifest: Optional[Dict[str, Any]] = None
    try:
        rule_payload = _extract_json_payload((run_dir / rule_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
        manifest_ref = str((rule_payload or {}).get("manifest", "")).strip()
        if manifest_ref:
            manifest_path = Path(manifest_ref)
            if not manifest_path.is_absolute():
                manifest_path = (ROOT / manifest_path).resolve()
            rule_manifest = _load_json(manifest_path)
    except Exception as exc:
        rule_payload = {"parse_error": str(exc)}
    steps["run_rule_cycle"] = _step_record("run_rule_cycle", rule_run, rule_payload)
    reports["rule_cycle"] = rule_manifest

    # 2) Strict gate
    strict_cmd = [
        python_exec,
        "scripts/eval_strict_pi_gate.py",
        "--profile",
        str(args.profile),
        "--holdout-jsonl",
        "tests/data/strict_pi_holdout/strict_pi_holdout_seed41.jsonl",
        "--seed",
        str(int(args.seed)),
    ]
    if bool(args.weekly_regression):
        strict_cmd.append("--weekly-regression")
    if args.strict_baseline_report:
        strict_cmd.extend(["--baseline-report", str(args.strict_baseline_report)])
    strict_run = _run_and_capture("eval_strict_pi_gate", strict_cmd, run_dir)
    commands.append(strict_run)
    strict_payload: Optional[Dict[str, Any]] = None
    strict_report: Optional[Dict[str, Any]] = None
    try:
        strict_payload = _extract_json_payload((run_dir / strict_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
        strict_report = strict_payload
    except Exception as exc:
        strict_payload = {"parse_error": str(exc)}
    steps["eval_strict_pi_gate"] = _step_record("eval_strict_pi_gate", strict_run, strict_payload)
    reports["strict_pi"] = strict_report

    # 3) Attachment eval
    attach_cmd = [
        python_exec,
        "scripts/eval_attachment_ingestion.py",
        "--profile",
        str(args.profile),
        "--seed",
        str(int(args.seed)),
        "--manifest",
        "tests/data/attachment_eval/manifest.jsonl",
    ]
    if bool(args.weekly_regression):
        attach_cmd.append("--weekly-regression")
    if args.attachment_baseline_report:
        attach_cmd.extend(["--baseline-report", str(args.attachment_baseline_report)])
    attach_run = _run_and_capture("eval_attachment_ingestion", attach_cmd, run_dir)
    commands.append(attach_run)
    attach_payload: Optional[Dict[str, Any]] = None
    attach_report: Optional[Dict[str, Any]] = None
    try:
        attach_payload = _extract_json_payload((run_dir / attach_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
        attach_report = attach_payload
    except Exception as exc:
        attach_payload = {"parse_error": str(exc)}
    steps["eval_attachment_ingestion"] = _step_record("eval_attachment_ingestion", attach_run, attach_payload)
    reports["attachment"] = attach_report

    # 4) Session benchmark (optional)
    session_payload: Optional[Dict[str, Any]] = None
    session_report: Optional[Dict[str, Any]] = None
    if bool(args.include_session_benchmark):
        session_cmd = [
            python_exec,
            "scripts/eval_session_pi_gate.py",
            "--profile",
            str(args.profile),
            "--mode",
            str(args.session_mode),
            "--pack",
            str(args.session_pack),
            "--seed",
            str(int(args.seed)),
        ]
        if bool(args.weekly_regression):
            session_cmd.append("--weekly-regression")
        if args.session_baseline_report:
            session_cmd.extend(["--baseline-report", str(args.session_baseline_report)])
        session_run = _run_and_capture("eval_session_pi_gate", session_cmd, run_dir)
        commands.append(session_run)
        try:
            session_payload = _extract_json_payload((run_dir / session_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
            session_report = session_payload
        except Exception as exc:
            session_payload = {"parse_error": str(exc)}
        steps["eval_session_pi_gate"] = _step_record("eval_session_pi_gate", session_run, session_payload)
    else:
        steps["eval_session_pi_gate"] = {
            "name": "eval_session_pi_gate",
            "status": "skipped",
            "reason": "disabled_by_flag",
            "payload": {"include_session_benchmark": False},
        }
    reports["session_benchmark"] = session_report

    # 5) BIPIA validation (real benchmark only when ready)
    bipia_payload: Optional[Dict[str, Any]] = None
    bipia_report: Optional[Dict[str, Any]] = None
    if bipia_ready:
        bipia_cmd = [
            python_exec,
            "scripts/run_bipia_validation.py",
            "--profile",
            str(args.profile),
            "--benchmark-root",
            str(args.bipia_benchmark_root),
            "--mode",
            str(args.bipia_mode),
            "--split",
            str(args.bipia_split),
            "--repeats",
            str(int(args.bipia_repeats)),
            "--seed-start",
            str(int(args.seed)),
        ]
        bipia_run = _run_and_capture("run_bipia_validation", bipia_cmd, run_dir)
        commands.append(bipia_run)
        try:
            bipia_payload = _extract_json_payload((run_dir / bipia_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
            run_id_bipia = str((bipia_payload or {}).get("run_id", "")).strip()
            if run_id_bipia:
                bipia_report_path = (ROOT / "artifacts" / "bipia_validation" / run_id_bipia / "stability_report.json").resolve()
                bipia_report = _load_json(bipia_report_path)
        except Exception as exc:
            bipia_payload = {"parse_error": str(exc)}
        steps["run_bipia_validation"] = _step_record("run_bipia_validation", bipia_run, bipia_payload)
    else:
        steps["run_bipia_validation"] = {
            "name": "run_bipia_validation",
            "status": "skipped",
            "reason": "not_ready_for_external_compare",
            "payload": {"qa_abstract_md5_ok": False},
        }
    reports["bipia"] = bipia_report

    # 6) External anchor: PINT
    pint_cmd = [
        python_exec,
        "scripts/eval_pint_omega.py",
        "--profile",
        str(args.profile),
        "--seed",
        str(int(args.seed)),
        "--dataset",
        str(args.pint_dataset),
    ]
    if bool(args.weekly_regression):
        pint_cmd.append("--weekly-regression")
    pint_run = _run_and_capture("eval_pint_omega", pint_cmd, run_dir)
    commands.append(pint_run)
    pint_payload: Optional[Dict[str, Any]] = None
    try:
        pint_payload = _extract_json_payload((run_dir / pint_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        pint_payload = {"parse_error": str(exc)}
    steps["eval_pint_omega"] = _step_record("eval_pint_omega", pint_run, pint_payload)
    reports["pint"] = pint_payload

    # 7) External anchor: WAInjectBench text subset
    wa_cmd = [
        python_exec,
        "scripts/eval_wainjectbench_text.py",
        "--profile",
        str(args.profile),
        "--seed",
        str(int(args.seed)),
        "--root",
        str(args.wainject_root),
    ]
    if int(args.wainject_max_samples) > 0:
        wa_cmd.extend(["--max-samples", str(int(args.wainject_max_samples))])
    if bool(args.weekly_regression):
        wa_cmd.append("--weekly-regression")
    wa_run = _run_and_capture("eval_wainjectbench_text", wa_cmd, run_dir)
    commands.append(wa_run)
    wa_payload: Optional[Dict[str, Any]] = None
    try:
        wa_payload = _extract_json_payload((run_dir / wa_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        wa_payload = {"parse_error": str(exc)}
    steps["eval_wainjectbench_text"] = _step_record("eval_wainjectbench_text", wa_run, wa_payload)
    reports["wainject"] = wa_payload

    # 8) Unified comparative report
    comp_cmd = [
        python_exec,
        "scripts/build_comparative_report.py",
        "--run-id",
        f"{run_id}_comparative",
    ]
    if isinstance(rule_payload, dict) and str(rule_payload.get("manifest", "")).strip():
        comp_cmd.extend(["--contour-manifest", _as_rel((ROOT / str(rule_payload["manifest"])).resolve())])
    if isinstance(pint_payload, dict) and isinstance(pint_payload.get("artifacts"), dict):
        comp_cmd.extend(["--pint-report", str(pint_payload["artifacts"].get("report_json", ""))])
    if isinstance(wa_payload, dict) and isinstance(wa_payload.get("artifacts"), dict):
        comp_cmd.extend(["--wainject-report", str(wa_payload["artifacts"].get("report_json", ""))])
    comp_run = _run_and_capture("build_comparative_report", comp_cmd, run_dir)
    commands.append(comp_run)
    comp_payload: Optional[Dict[str, Any]] = None
    comp_report: Optional[Dict[str, Any]] = None
    try:
        comp_payload = _extract_json_payload((run_dir / comp_run.stdout_file).read_text(encoding="utf-8", errors="replace"))
        comp_report_path = Path(str((comp_payload.get("artifacts", {}) or {}).get("report_json", "")))
        if not comp_report_path.is_absolute():
            comp_report_path = (ROOT / comp_report_path).resolve()
        comp_report = _load_json(comp_report_path)
    except Exception as exc:
        comp_payload = {"parse_error": str(exc)}
    steps["build_comparative_report"] = _step_record("build_comparative_report", comp_run, comp_payload)
    reports["comparative"] = comp_report

    internal_ok = (
        str((strict_report or {}).get("status", "unknown")) in {"ok"}
        and str((attach_report or {}).get("status", "unknown")) in {"ok"}
        and (not bool(args.include_session_benchmark) or str((session_report or {}).get("status", "unknown")) in {"ok"})
        and int(rule_run.exit_code) == 0
    )
    comparative_ok = int(comp_run.exit_code) == 0 and isinstance(comp_report, dict)
    status = "OK" if internal_ok and comparative_ok else "FAILED"

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "profile": str(args.profile),
        "seed": int(args.seed),
        "weekly_regression": bool(args.weekly_regression),
        "config_refs": config_refs_from_snapshot(snapshot, code_commit="local"),
        "readiness": {
            "bipia_real_data_ready": bool(bipia_ready),
            "bipia_external_compare_status": "ready" if bipia_ready else "not_ready_for_external_compare",
        },
        "steps": steps,
        "reports": reports,
        "artifacts": {
            "manifest_json": _as_rel(run_dir / "manifest.json"),
            "run_dir": _as_rel(run_dir),
            "comparative_report_json": str(((comp_payload or {}).get("artifacts", {}) or {}).get("report_json")) if isinstance(comp_payload, dict) else None,
            "comparative_report_md": str(((comp_payload or {}).get("artifacts", {}) or {}).get("report_md")) if isinstance(comp_payload, dict) else None,
        },
        "commands": [asdict(c) for c in commands],
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    latest_path = (ROOT / str(args.artifacts_root) / "LATEST.json").resolve()
    latest_path.write_text(
        json.dumps({"run_id": run_id, "status": status, "manifest": _as_rel(manifest_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "run_id": run_id,
                "status": status,
                "manifest": _as_rel(manifest_path),
                "latest": _as_rel(latest_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
