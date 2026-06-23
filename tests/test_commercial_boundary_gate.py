from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_static_docker_boundary_gate_script() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, "scripts/check_docker_image_hygiene.py"], cwd=str(root), check=True)
