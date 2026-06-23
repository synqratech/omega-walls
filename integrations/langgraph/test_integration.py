from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SMOKE_SCRIPT = "scripts/smoke_langgraph_guard.py"


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    proc = subprocess.run(
        [sys.executable, SMOKE_SCRIPT, "--strict"],
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
