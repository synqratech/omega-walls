from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_run_eval_requires_semantic_and_fails_on_missing_model():
    root = Path(__file__).resolve().parent.parent
    env = dict(os.environ)
    env["BIPIA_COMMIT"] = "deadbeef"

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/run_eval.py",
            "--whitebox-max-samples",
            "5",
            "--require-semantic",
            "--semantic-model-path",
            "missing_semantic_model_dir_12345",
        ],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode != 0
    text = f"{proc.stdout}\n{proc.stderr}".lower()
    assert "semantic projector" in text
