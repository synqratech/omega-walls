#!/usr/bin/env python3
"""Build physically separated Omega Walls wheels from one monorepo.

Usage:
  python scripts/build_edition_wheels.py --edition community
  python scripts/build_edition_wheels.py --edition enterprise
  python scripts/build_edition_wheels.py --edition all
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_ENTERPRISE_STAGE_FILES = (
    "README_ENTERPRISE_PYPI.md",
    "pyproject.enterprise.toml",
)
_ENTERPRISE_STAGE_DIRS = (
    "omega_walls_enterprise",
)


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _clean_build_state(cwd: Path) -> None:
    for pattern in ("build", "*.egg-info"):
        for path in cwd.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()


def _build_wheel(cwd: Path, outdir: Path) -> None:
    _clean_build_state(cwd)
    try:
        _run([sys.executable, "-m", "build", "--wheel", "--outdir", str(outdir)], cwd=cwd)
    except subprocess.CalledProcessError as exc:
        # Minimal environments may not have the `build` package installed.
        # `pip wheel --no-deps .` still exercises setuptools packaging and is
        # enough for boundary verification in CI smoke jobs.
        if exc.returncode != 1:
            raise
        _run([sys.executable, "-m", "pip", "wheel", "--no-build-isolation", "--no-deps", ".", "-w", str(outdir)], cwd=cwd)


def build_community(outdir: Path) -> None:
    _build_wheel(ROOT, outdir)


def build_enterprise(outdir: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="omega-enterprise-build-") as td:
        work = Path(td) / "src"
        work.mkdir(parents=True, exist_ok=True)
        for name in _ENTERPRISE_STAGE_FILES:
            shutil.copy2(ROOT / name, work / name)
        for name in _ENTERPRISE_STAGE_DIRS:
            shutil.copytree(ROOT / name, work / name)
        shutil.copy2(work / "pyproject.enterprise.toml", work / "pyproject.toml")
        (work / "MANIFEST.in").write_text(
            "include pyproject.toml\n"
            "include README_ENTERPRISE_PYPI.md\n"
            "recursive-include omega_walls_enterprise *.py *.json *.yml *.yaml *.md\n"
            "prune omega\n"
            "prune tests\n"
            "prune docs\n"
            "prune artifacts\n"
            "prune data\n"
            "prune internal_data\n"
            "global-exclude __pycache__ *.py[cod] .DS_Store\n",
            encoding="utf-8",
        )
        _build_wheel(work, outdir)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edition", choices=["community", "enterprise", "all"], default="all")
    parser.add_argument("--outdir", default="dist")
    args = parser.parse_args()
    outdir = (ROOT / args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.edition in {"community", "all"}:
        build_community(outdir)
    if args.edition in {"enterprise", "all"}:
        build_enterprise(outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
