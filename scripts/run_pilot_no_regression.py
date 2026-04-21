from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

ROOT = Path(__file__).resolve().parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run(name: str, argv: List[str], out_dir: Path) -> dict[str, Any]:
    proc = subprocess.run(argv, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    stdout = out_dir / f"{name}.stdout.txt"
    stderr = out_dir / f"{name}.stderr.txt"
    stdout.write_text(proc.stdout, encoding="utf-8")
    stderr.write_text(proc.stderr, encoding="utf-8")
    return {
        "name": name,
        "argv": argv,
        "exit_code": int(proc.returncode),
        "stdout": str(stdout.relative_to(ROOT).as_posix()),
        "stderr": str(stderr.relative_to(ROOT).as_posix()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run no-regression gate for dev/local behaviors")
    parser.add_argument("--output-root", default="artifacts/pilot_no_regression")
    args = parser.parse_args()

    out_dir = ROOT / args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    commands = [
        (
            "profiles_isolation",
            [
                py,
                "-m",
                "pytest",
                "-q",
                "tests/test_profile_isolation_control_outcome.py",
                "tests/test_tool_gateway.py",
                "tests/test_off_policy.py",
            ],
        ),
        (
            "integration_smoke",
            [
                py,
                "-m",
                "pytest",
                "-q",
                "tests/test_integration_harness.py",
            ],
        ),
        (
            "replay_contracts",
            [
                py,
                "-m",
                "pytest",
                "-q",
                "tests/test_replay_incident.py",
            ],
        ),
    ]
    runs = [_run(name, argv, out_dir) for name, argv in commands]
    failed = [r for r in runs if int(r["exit_code"]) != 0]
    summary = {
        "event": "pilot_no_regression_gate_v1",
        "created_utc": _utc_now(),
        "status": "ok" if not failed else "fail",
        "runs": runs,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
