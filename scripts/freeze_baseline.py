from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import config_refs_from_snapshot, load_resolved_config


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _pick_python() -> str:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


@dataclass
class CommandRun:
    name: str
    argv: List[str]
    exit_code: int
    duration_sec: float
    stdout_file: str
    stderr_file: str


def _run_and_capture(name: str, argv: List[str], out_dir: Path) -> CommandRun:
    start = time.time()
    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    duration = round(time.time() - start, 3)
    stdout_file = out_dir / f"{name}.stdout.txt"
    stderr_file = out_dir / f"{name}.stderr.txt"
    stdout_file.write_text(proc.stdout, encoding="utf-8")
    stderr_file.write_text(proc.stderr, encoding="utf-8")
    return CommandRun(
        name=name,
        argv=argv,
        exit_code=int(proc.returncode),
        duration_sec=duration,
        stdout_file=str(stdout_file.relative_to(out_dir.parent).as_posix()),
        stderr_file=str(stderr_file.relative_to(out_dir.parent).as_posix()),
    )


def _copy_config_snapshot(snapshot_file_hashes: Dict[str, str], baseline_dir: Path) -> Dict[str, str]:
    copied: Dict[str, str] = {}
    for cfg_path, digest in snapshot_file_hashes.items():
        src = ROOT / Path(cfg_path)
        dst = baseline_dir / "config_snapshot" / Path(cfg_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied[str(Path(cfg_path).as_posix())] = digest
    return copied


def _script_hashes(paths: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in paths:
        path = ROOT / p
        if path.exists():
            out[p] = _sha256_file(path)
    return out


def _try_parse_json_from_text(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def _extract_tool_gateway_metrics(command_runs: List[CommandRun], baseline_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for run in command_runs:
        stdout_path = baseline_dir / f"{run.name}.stdout.txt"
        if not stdout_path.exists():
            continue
        payload = _try_parse_json_from_text(stdout_path.read_text(encoding="utf-8", errors="replace"))
        if not isinstance(payload, dict):
            continue

        if run.name == "run_local_contour":
            summary = payload.get("summary", {})
            tgc = summary.get("tool_gateway_coverage", {})
            if isinstance(tgc, dict):
                out["run_local_contour"] = {
                    "events": int(tgc.get("events", 0)),
                    "requests_seen": int(tgc.get("requests_seen", 0)),
                    "coverage": float(tgc.get("coverage", 1.0)),
                    "orphan_executions": int(tgc.get("orphan_executions", 0)),
                }
        if run.name == "smoke_real_rag":
            reports = payload.get("reports", [])
            if isinstance(reports, list):
                total_events = 0
                total_requests = 0
                for report in reports:
                    if not isinstance(report, dict):
                        continue
                    total_events += int(len(report.get("tool_gateway_events", [])))
                    total_requests += int(len(report.get("inferred_tool_requests", [])))
                out["smoke_real_rag"] = {
                    "events": total_events,
                    "requests_seen": total_requests,
                    "coverage": (float(total_events) / float(total_requests) if total_requests else 1.0),
                }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze reproducible baseline artifact")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--model-path", default=".")
    parser.add_argument("--output-root", default="artifacts/release_baseline")
    parser.add_argument("--label", default="before")
    parser.add_argument("--enforce-whitebox", action="store_true")
    parser.add_argument("--whitebox-max-samples", type=int, default=200)
    args = parser.parse_args()

    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    snapshot = load_resolved_config(profile=args.profile)
    resolved_sha_short = snapshot.resolved_sha256[:12]
    utc_now = datetime.now(timezone.utc)
    run_id = f"{args.label}_{utc_now.strftime('%Y%m%dT%H%M%SZ')}_{resolved_sha_short}"
    baseline_dir = output_root / run_id
    baseline_dir.mkdir(parents=True, exist_ok=True)

    python_exec = _pick_python()

    eval_cmd = [
        python_exec,
        "scripts/run_eval.py",
        "--whitebox-max-samples",
        str(args.whitebox_max_samples),
    ]
    if args.enforce_whitebox:
        eval_cmd.append("--enforce-whitebox")

    commands = [
        (
            "run_eval",
            eval_cmd,
        ),
        (
            "run_local_contour",
            [
                python_exec,
                "scripts/run_local_contour.py",
                "--llm-backend",
                "local",
                "--model-path",
                args.model_path,
                "--profile",
                args.profile,
                "--strict",
                "--output",
                str((baseline_dir / "local_contour_report.json").as_posix()),
            ],
        ),
        (
            "smoke_real_rag",
            [
                python_exec,
                "scripts/smoke_real_rag.py",
                "--profile",
                args.profile,
                "--model-path",
                args.model_path,
                "--strict",
            ],
        ),
        (
            "pytest",
            [
                python_exec,
                "-m",
                "pytest",
                "-q",
            ],
        ),
    ]

    command_runs: List[CommandRun] = []
    any_fail = False
    for name, argv in commands:
        result = _run_and_capture(name=name, argv=argv, out_dir=baseline_dir)
        command_runs.append(result)
        if result.exit_code != 0:
            any_fail = True

    copied_configs = _copy_config_snapshot(snapshot.file_hashes, baseline_dir)
    (baseline_dir / "resolved_config.json").write_text(
        json.dumps(snapshot.resolved, ensure_ascii=True, indent=2, default=str),
        encoding="utf-8",
    )

    seeds = {
        "run_eval": {
            "generator_seed": 7,
            "whitebox_seed": 19,
            "whitebox_max_samples": args.whitebox_max_samples,
            "whitebox_beam_width": 4,
            "whitebox_max_iters": 5,
            "whitebox_mutations": 3,
            "enforce_whitebox": args.enforce_whitebox,
        },
        "run_local_contour": {
            "deterministic_scenario_set": "SCENARIOS in scripts/run_local_contour.py",
            "llm_temperature": 0.0,
        },
        "smoke_real_rag": {
            "deterministic_scenario_set": "DEFAULT_SCENARIOS in scripts/smoke_real_rag.py",
            "llm_temperature": 0.0,
        },
        "pytest": {
            "note": "No additional random seed override used in command.",
        },
    }

    script_hashes = _script_hashes(
        [
            "scripts/run_eval.py",
            "scripts/run_release_gate.py",
            "scripts/prepare_bipia_contexts.py",
            "scripts/run_bipia_validation.py",
            "scripts/run_local_contour.py",
            "scripts/smoke_real_rag.py",
            "scripts/freeze_baseline.py",
            "scripts/run_all_checks.ps1",
            "scripts/replay_incident.py",
            "scripts/ops_state_recovery.py",
            "redteam/whitebox_optimizer.py",
            "omega/eval/bipia_adapter.py",
            "omega/eval/bipia_manifest.py",
            "omega/eval/bipia_metrics.py",
            "omega/rag/retriever_prod_adapter.py",
            "omega/rag/retriever_provider.py",
            "schemas/incident_artifact_v1.json",
            "schemas/tool_gateway_step_v1.json",
        ]
    )
    integration_metrics = _extract_tool_gateway_metrics(command_runs=command_runs, baseline_dir=baseline_dir)

    manifest = {
        "run_id": run_id,
        "created_utc": utc_now.isoformat(),
        "profile": args.profile,
        "python_exec": python_exec,
        "platform": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "config_refs": config_refs_from_snapshot(snapshot, code_commit="local"),
        "config_hashes": copied_configs,
        "script_hashes": script_hashes,
        "seed_set": seeds,
        "integration_metrics": integration_metrics,
        "commands": [asdict(c) for c in command_runs],
        "status": "FAILED" if any_fail else "OK",
    }

    manifest_path = baseline_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    latest_path = output_root / "LATEST.json"
    latest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "status": manifest["status"],
                "manifest": str(manifest_path.relative_to(ROOT).as_posix()),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"run_id": run_id, "status": manifest["status"], "manifest": str(manifest_path)}, indent=2))
    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
