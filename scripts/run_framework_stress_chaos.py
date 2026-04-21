from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run stress/chaos layer for framework matrix stand")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--artifacts-root", default="artifacts")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    cmd = [
        sys.executable,
        str((ROOT / "scripts" / "run_framework_matrix_stand.py").resolve()),
        "--layer",
        "stress",
        "--profile",
        str(args.profile),
        "--artifacts-root",
        str(args.artifacts_root),
    ]
    if bool(args.strict):
        cmd.append("--strict")
    if args.run_id:
        cmd.extend(["--run-id", str(args.run_id)])

    proc = subprocess.run(cmd, cwd=str(ROOT))  # noqa: S603
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
