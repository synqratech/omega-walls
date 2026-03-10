from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _now_utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_smoke(name: str, argv: list[str], out_dir: Path) -> dict[str, Any]:
    proc = subprocess.run(
        argv,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
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
    parser = argparse.ArgumentParser(description="Run framework integration smokes (LangChain + LlamaIndex)")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    run_dir = Path(args.output_dir) if args.output_dir else ROOT / "artifacts" / "framework_smoke" / _now_utc_stamp()
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    langchain_report_path = run_dir / "langchain_report.json"
    llamaindex_report_path = run_dir / "llamaindex_report.json"

    command_runs = [
        _run_smoke(
            "smoke_langchain",
            [
                py,
                "scripts/smoke_langchain.py",
                "--profile",
                args.profile,
                "--top-k",
                str(args.top_k),
                "--output",
                str(langchain_report_path),
                *(["--strict"] if args.strict else []),
            ],
            run_dir,
        ),
        _run_smoke(
            "smoke_llamaindex",
            [
                py,
                "scripts/smoke_llamaindex.py",
                "--profile",
                args.profile,
                "--top-k",
                str(args.top_k),
                "--output",
                str(llamaindex_report_path),
                *(["--strict"] if args.strict else []),
            ],
            run_dir,
        ),
    ]

    reports: dict[str, Any] = {}
    for path, key in ((langchain_report_path, "langchain"), (llamaindex_report_path, "llamaindex")):
        if path.exists():
            reports[key] = _load_json(path)
        else:
            reports[key] = {"framework": key, "failures": [f"missing report: {path}"], "summary": {}}

    total_failures = 0
    min_gateway_coverage = 1.0
    total_orphans = 0
    frameworks_summary: dict[str, Any] = {}
    for key, report in reports.items():
        failures = report.get("failures", [])
        summary = report.get("summary", {}) if isinstance(report, dict) else {}
        failure_count = len(failures) if isinstance(failures, list) else 1
        coverage = float(summary.get("gateway_coverage", 0.0))
        orphans = int(summary.get("orphan_executions", 0))
        frameworks_summary[key] = {
            "failure_count": failure_count,
            "gateway_coverage": coverage,
            "orphan_executions": orphans,
        }
        total_failures += failure_count
        min_gateway_coverage = min(min_gateway_coverage, coverage)
        total_orphans += orphans

    non_zero_exits = [r for r in command_runs if int(r["exit_code"]) != 0]
    overall_ok = (not non_zero_exits) and total_failures == 0 and min_gateway_coverage >= 1.0 and total_orphans == 0

    summary = {
        "status": "ok" if overall_ok else "fail",
        "profile": args.profile,
        "strict": bool(args.strict),
        "run_dir": str(run_dir),
        "frameworks": frameworks_summary,
        "metrics": {
            "total_failures": total_failures,
            "min_gateway_coverage": min_gateway_coverage,
            "total_orphans": total_orphans,
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
