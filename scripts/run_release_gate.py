from __future__ import annotations

import argparse
import json
import os
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


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return CommandRun(
        name=name,
        argv=argv,
        exit_code=int(proc.returncode),
        duration_sec=duration,
        stdout_file=str(stdout_path),
        stderr_file=str(stderr_path),
    )


def _try_parse_json_from_text(text: str) -> Any:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def _load_json_file(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _extract_run_local_contour_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", {}) if isinstance(payload, dict) else {}
    scenarios = payload.get("scenarios", []) if isinstance(payload, dict) else []

    benign_off_count = 0
    steps_to_off_sla_violations = 0
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            continue
        expected_off = bool(scenario.get("expected_off", False))
        final_off = bool(scenario.get("final_off", False))
        steps_to_off = scenario.get("steps_to_off")
        max_steps_to_off = scenario.get("max_steps_to_off")
        if (not expected_off) and final_off:
            benign_off_count += 1
        if expected_off and max_steps_to_off is not None:
            if steps_to_off is None or int(steps_to_off) > int(max_steps_to_off):
                steps_to_off_sla_violations += 1

    tool_freeze = summary.get("tool_freeze_coverage", {}) if isinstance(summary, dict) else {}
    tool_gateway = summary.get("tool_gateway_coverage", {}) if isinstance(summary, dict) else {}

    return {
        "fp_horizon_step": summary.get("fp_horizon_step"),
        "benign_off_count": benign_off_count,
        "steps_to_off_sla_violations": steps_to_off_sla_violations,
        "tool_freeze_enforcement_rate": float(tool_freeze.get("rate", 0.0)),
        "tool_gateway_coverage": float(tool_gateway.get("coverage", 0.0)),
        "orphan_tool_executions": int(tool_gateway.get("orphan_executions", 0)),
    }


def _extract_smoke_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    reports = payload.get("reports", []) if isinstance(payload, dict) else []
    failures = payload.get("failures", []) if isinstance(payload, dict) else []

    gateway_values: List[float] = []
    bypass_count = 0
    for report in reports:
        if not isinstance(report, dict):
            continue
        gateway_values.append(float(report.get("gateway_coverage", 0.0)))
        decisions = report.get("tool_decisions", [])
        executions = report.get("tool_executions", [])
        for dec, exe in zip(decisions, executions):
            allowed = bool(dec.get("allowed", False)) if isinstance(dec, dict) else False
            executed = bool(exe.get("executed", False)) if isinstance(exe, dict) else False
            if executed and not allowed:
                bypass_count += 1

    coverage_min = min(gateway_values) if gateway_values else 1.0
    coverage_avg = (sum(gateway_values) / len(gateway_values)) if gateway_values else 1.0
    return {
        "failures_count": len(failures) if isinstance(failures, list) else 1,
        "gateway_coverage_min": float(coverage_min),
        "gateway_coverage_avg": float(coverage_avg),
        "bypass_count": int(bypass_count),
    }


def get_metric_by_path(payload: Dict[str, Any], metric_path: str) -> Any:
    cur: Any = payload
    for part in metric_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def evaluate_gates(metrics: Dict[str, Any], gates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for gate in gates:
        gate_id = str(gate.get("id", "UNKNOWN"))
        metric_path = str(gate.get("metric", ""))
        op = str(gate.get("op", "")).lower()
        threshold = gate.get("threshold")
        fail_message = str(gate.get("fail_message", "gate failed"))
        evidence_field = str(gate.get("evidence_field", metric_path))
        observed = get_metric_by_path(metrics, metric_path)

        passed = False
        if op == "eq":
            if isinstance(observed, (int, float)) and isinstance(threshold, (int, float)):
                passed = abs(float(observed) - float(threshold)) <= 1e-9
            else:
                passed = observed == threshold
        elif op == "ge":
            obs_f = _to_float(observed)
            thr_f = _to_float(threshold)
            passed = bool(obs_f is not None and thr_f is not None and obs_f >= thr_f)
        elif op == "le":
            obs_f = _to_float(observed)
            thr_f = _to_float(threshold)
            passed = bool(obs_f is not None and thr_f is not None and obs_f <= thr_f)
        elif op == "is_null":
            passed = observed is None
        elif op == "not_null":
            passed = observed is not None

        out.append(
            {
                "id": gate_id,
                "status": "PASS" if passed else "FAIL",
                "metric": metric_path,
                "operator": op,
                "threshold": threshold,
                "observed": observed,
                "fail_message": fail_message,
                "evidence_field": evidence_field,
                "evidence_value": get_metric_by_path(metrics, evidence_field),
            }
        )
    return out


def _render_markdown_report(
    run_id: str,
    status: str,
    profile: str,
    created_utc: str,
    gates: List[Dict[str, Any]],
    metrics_path: str,
) -> str:
    lines = [
        "# Release Gate Report",
        "",
        f"- run_id: `{run_id}`",
        f"- status: **{status}**",
        f"- profile: `{profile}`",
        f"- created_utc: `{created_utc}`",
        f"- metrics: `{metrics_path}`",
        "",
        "| Gate ID | Status | Metric | Operator | Threshold | Observed |",
        "|---|---|---|---|---:|---:|",
    ]
    for gate in gates:
        lines.append(
            f"| {gate['id']} | {gate['status']} | `{gate['metric']}` | `{gate['operator']}` | "
            f"`{gate['threshold']}` | `{gate['observed']}` |"
        )
    failures = [g for g in gates if g["status"] != "PASS"]
    lines.append("")
    if failures:
        lines.append("## Failures")
        for gate in failures:
            lines.append(f"- `{gate['id']}`: {gate['fail_message']}")
    else:
        lines.append("All gates passed.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Machine-checked release gate with go/no-go output")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--artifacts-root", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved
    rg_cfg = cfg.get("release_gate", {})
    artifacts_root = Path(args.artifacts_root or rg_cfg.get("artifacts_root", "artifacts/release_gate"))
    run_id = f"release_gate_{_utc_now().strftime('%Y%m%dT%H%M%SZ')}_{snapshot.resolved_sha256[:12]}"
    run_dir = (ROOT / artifacts_root / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    python_exec = sys.executable
    commands_cfg = rg_cfg.get("commands", {})

    pytest_args = commands_cfg.get("pytest_args", ["-m", "pytest", "-q"])
    eval_cfg = commands_cfg.get("run_eval", {})
    contour_cfg = commands_cfg.get("run_local_contour", {})
    smoke_cfg = commands_cfg.get("smoke_real_rag", {})

    contour_output_path = run_dir / "local_contour_report.json"
    commands: List[Tuple[str, List[str]]] = [
        ("pytest", [python_exec, *pytest_args]),
        (
            "run_eval",
            [
                python_exec,
                "scripts/run_eval.py",
                "--whitebox-max-samples",
                str(int(eval_cfg.get("whitebox_max_samples", 200))),
                *(["--enforce-whitebox"] if bool(eval_cfg.get("enforce_whitebox", True)) else []),
                *(["--require-semantic"] if bool(eval_cfg.get("require_semantic", False)) else []),
                *(["--enforce-bipia"] if bool(eval_cfg.get("enforce_bipia", False)) else []),
                *(["--semantic-model-path", str(eval_cfg.get("semantic_model_path"))] if "semantic_model_path" in eval_cfg else []),
                *(["--bipia-mode", str(eval_cfg.get("bipia_mode", "sampled"))] if "bipia_mode" in eval_cfg else []),
                *(["--bipia-split", str(eval_cfg.get("bipia_split", "test"))] if "bipia_split" in eval_cfg else []),
                *(
                    [
                        "--bipia-max-contexts-per-task",
                        str(int(eval_cfg.get("bipia_max_contexts_per_task", 20))),
                    ]
                    if "bipia_max_contexts_per_task" in eval_cfg
                    else []
                ),
                *(["--bipia-seed", str(int(eval_cfg.get("bipia_seed", 41)))] if "bipia_seed" in eval_cfg else []),
                *(["--bipia-benchmark-root", str(eval_cfg.get("bipia_benchmark_root"))] if "bipia_benchmark_root" in eval_cfg else []),
            ],
        ),
        (
            "run_local_contour",
            [
                python_exec,
                "scripts/run_local_contour.py",
                "--llm-backend",
                str(contour_cfg.get("llm_backend", "mock")),
                "--profile",
                args.profile,
                "--fp-steps",
                str(int(contour_cfg.get("fp_steps", 40))),
                "--output",
                str(contour_output_path),
                *(["--strict"] if bool(contour_cfg.get("strict", True)) else []),
            ],
        ),
        (
            "smoke_real_rag",
            [
                python_exec,
                "scripts/smoke_real_rag.py",
                "--profile",
                args.profile,
                "--llm-backend",
                str(smoke_cfg.get("llm_backend", "mock")),
                *(["--strict"] if bool(smoke_cfg.get("strict", True)) else []),
            ],
        ),
    ]

    command_runs: List[CommandRun] = []
    for name, argv in commands:
        command_runs.append(_run_and_capture(name=name, argv=argv, out_dir=run_dir))

    command_map: Dict[str, Dict[str, Any]] = {}
    for run in command_runs:
        command_map[run.name] = {
            "exit_code": run.exit_code,
            "duration_sec": run.duration_sec,
            "stdout_file": run.stdout_file,
            "stderr_file": run.stderr_file,
        }

    run_eval_payload = _try_parse_json_from_text((run_dir / "run_eval.stdout.txt").read_text(encoding="utf-8"))
    contour_payload = _load_json_file(contour_output_path)
    smoke_payload = _try_parse_json_from_text((run_dir / "smoke_real_rag.stdout.txt").read_text(encoding="utf-8"))

    release_metrics = {
        "run_id": run_id,
        "created_utc": _utc_now().isoformat(),
        "profile": args.profile,
        "config_refs": config_refs_from_snapshot(snapshot, code_commit="local"),
        "commands": command_map,
        "run_eval": run_eval_payload if isinstance(run_eval_payload, dict) else {},
        "run_local_contour": _extract_run_local_contour_metrics(contour_payload if isinstance(contour_payload, dict) else {}),
        "smoke_real_rag": _extract_smoke_metrics(smoke_payload if isinstance(smoke_payload, dict) else {}),
        "evidence": {
            "run_eval_stdout": str((run_dir / "run_eval.stdout.txt").relative_to(ROOT).as_posix()),
            "run_local_contour_report": str(contour_output_path.relative_to(ROOT).as_posix()),
            "smoke_real_rag_stdout": str((run_dir / "smoke_real_rag.stdout.txt").relative_to(ROOT).as_posix()),
            "pytest_stdout": str((run_dir / "pytest.stdout.txt").relative_to(ROOT).as_posix()),
        },
    }

    gates_cfg = rg_cfg.get("gates", [])
    gate_results = evaluate_gates(release_metrics, gates_cfg if isinstance(gates_cfg, list) else [])
    status = "GO" if all(g["status"] == "PASS" for g in gate_results) else "NO_GO"

    metrics_path = run_dir / "release_metrics.json"
    metrics_path.write_text(json.dumps(release_metrics, ensure_ascii=True, indent=2), encoding="utf-8")

    go_nogo = {
        "run_id": run_id,
        "created_utc": release_metrics["created_utc"],
        "profile": args.profile,
        "status": status,
        "gates": gate_results,
        "failed_gate_ids": [g["id"] for g in gate_results if g["status"] != "PASS"],
        "metrics_file": str(metrics_path.relative_to(ROOT).as_posix()),
        "config_refs": release_metrics["config_refs"],
    }
    go_nogo_json_path = run_dir / "go_nogo_report.json"
    go_nogo_json_path.write_text(json.dumps(go_nogo, ensure_ascii=True, indent=2), encoding="utf-8")

    md = _render_markdown_report(
        run_id=run_id,
        status=status,
        profile=args.profile,
        created_utc=release_metrics["created_utc"],
        gates=gate_results,
        metrics_path=str(metrics_path.relative_to(ROOT).as_posix()),
    )
    go_nogo_md_path = run_dir / "go_nogo_report.md"
    go_nogo_md_path.write_text(md, encoding="utf-8")

    history_path = (ROOT / artifacts_root / "history.jsonl")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_event = {
        "run_id": run_id,
        "created_utc": release_metrics["created_utc"],
        "profile": args.profile,
        "status": status,
        "failed_gate_ids": go_nogo["failed_gate_ids"],
        "metrics_file": str(metrics_path.relative_to(ROOT).as_posix()),
        "report_file": str(go_nogo_json_path.relative_to(ROOT).as_posix()),
    }
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(history_event, ensure_ascii=True) + "\n")

    print(json.dumps(go_nogo, ensure_ascii=True, indent=2))
    if args.strict and status != "GO":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
