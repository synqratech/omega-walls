from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_tmp_root() -> Path:
    env_candidates = [os.environ.get("OMEGA_TMP_ROOT"), os.environ.get("TMP"), os.environ.get("TEMP")]
    for candidate in env_candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".omega_tmp_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return path
        except Exception:
            continue
    if os.name == "nt":
        win_tmp = Path("C:/tmp")
        try:
            win_tmp.mkdir(parents=True, exist_ok=True)
            probe = win_tmp / ".omega_tmp_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return win_tmp
        except Exception:
            pass
    return Path(tempfile.gettempdir())


def _run_smoke(name: str, argv: list[str], out_dir: Path, *, temp_root: Path) -> dict[str, Any]:
    env = dict(os.environ)
    env["TMP"] = str(temp_root)
    env["TEMP"] = str(temp_root)
    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    stdout_path = out_dir / f"{name}.stdout.txt"
    stderr_path = out_dir / f"{name}.stderr.txt"
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "argv": argv,
        "exit_code": int(proc.returncode),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run framework guard smokes (LangChain/LangGraph/LlamaIndex/Haystack/AutoGen/CrewAI)")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--require-pytest", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "framework_smoke" / _now_utc_stamp()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    temp_root = _resolve_tmp_root()
    preflight = {
        "pytest_available": bool(importlib.util.find_spec("pytest") is not None),
        "temp_root": str(temp_root),
    }
    smoke_specs = [
        ("langchain_guard", "scripts/smoke_langchain_guard.py"),
        ("langgraph_guard", "scripts/smoke_langgraph_guard.py"),
        ("llamaindex_guard", "scripts/smoke_llamaindex_guard.py"),
        ("haystack_guard", "scripts/smoke_haystack_guard.py"),
        ("autogen_guard", "scripts/smoke_autogen_guard.py"),
        ("crewai_guard", "scripts/smoke_crewai_guard.py"),
    ]

    command_runs = []
    report_paths: dict[str, Path] = {}
    for key, script_path in smoke_specs:
        report_path = run_dir / f"{key}_report.json"
        report_paths[key] = report_path
        command_runs.append(
            _run_smoke(
                f"smoke_{key}",
                [
                    py,
                    script_path,
                    "--profile",
                    args.profile,
                    "--output",
                    str(report_path),
                    *(["--strict"] if args.strict else []),
                ],
                run_dir,
                temp_root=temp_root,
            )
        )

    reports: dict[str, Any] = {}
    for key, path in report_paths.items():
        if path.exists():
            reports[key] = _load_json(path)
        else:
            reports[key] = {"framework": key, "failures": [f"missing report: {path}"], "summary": {}}

    total_failures = 0
    min_gateway_coverage = 1.0
    total_orphans = 0
    structured_contract_ok = True
    security_metadata_ok = True
    frameworks_summary: dict[str, Any] = {}
    for key, report in reports.items():
        failures = report.get("failures", [])
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        failure_count = len(failures) if isinstance(failures, list) else 1
        coverage = float(summary.get("gateway_coverage", 0.0))
        orphans = int(summary.get("orphan_executions", 0))
        block_ok = bool(summary.get("structured_block_contract_ok", False))
        secmeta_ok = bool(summary.get("security_metadata_present", False))
        frameworks_summary[key] = {
            "failure_count": failure_count,
            "gateway_coverage": coverage,
            "orphan_executions": orphans,
            "structured_block_contract_ok": block_ok,
            "security_metadata_present": secmeta_ok,
        }
        total_failures += failure_count
        min_gateway_coverage = min(min_gateway_coverage, coverage)
        total_orphans += orphans
        structured_contract_ok = structured_contract_ok and block_ok
        security_metadata_ok = security_metadata_ok and secmeta_ok

    non_zero_exits = [r for r in command_runs if int(r["exit_code"]) != 0]
    preflight_errors: list[str] = []
    if bool(args.require_pytest) and not preflight["pytest_available"]:
        preflight_errors.append(
            "pytest is not installed. Install test deps with: python -m pip install -e \".[dev,integrations]\""
        )
    overall_ok = (
        (not non_zero_exits)
        and total_failures == 0
        and min_gateway_coverage >= 1.0
        and total_orphans == 0
        and structured_contract_ok
        and security_metadata_ok
        and (not preflight_errors)
    )

    summary = {
        "status": "ok" if overall_ok else "fail",
        "profile": args.profile,
        "strict": bool(args.strict),
        "run_dir": str(run_dir),
        "framework_count": len(smoke_specs),
        "frameworks": frameworks_summary,
        "preflight": preflight,
        "preflight_errors": preflight_errors,
        "metrics": {
            "total_failures": total_failures,
            "min_gateway_coverage": min_gateway_coverage,
            "total_orphans": total_orphans,
            "structured_block_contract_ok": bool(structured_contract_ok),
            "security_metadata_ok": bool(security_metadata_ok),
        },
        "command_runs": command_runs,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if args.strict and not overall_ok:
        return 1
    return 0 if not non_zero_exits else 1


if __name__ == "__main__":
    raise SystemExit(main())
