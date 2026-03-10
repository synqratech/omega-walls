from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_eval_enforce_whitebox_thresholds_active():
    root = Path(__file__).resolve().parent.parent
    proc = subprocess.run(
        [sys.executable, "scripts/run_eval.py", "--enforce-whitebox", "--whitebox-max-samples", "5"],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode != 0
    assert "whitebox evaluated < 200" in (proc.stdout + proc.stderr)
