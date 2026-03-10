from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_prepare_bipia_contexts_passes_on_ready_fixture():
    root = Path(__file__).resolve().parent.parent
    tmp_dir = _mk_local_tmp("bipia-prepare-pass")
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_bipia_contexts.py",
            "--benchmark-root",
            "tests/data/bipia_fixture/benchmark",
            "--strict",
            "--artifacts-root",
            str((tmp_dir / "artifacts").as_posix()),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0
    payload = json.loads(proc.stdout)
    report_path = Path(payload["report"])
    assert report_path.exists()


def test_prepare_bipia_contexts_fails_when_qa_missing_and_no_newsqa():
    root = Path(__file__).resolve().parent.parent
    src = root / "tests" / "data" / "bipia_fixture" / "benchmark"
    tmp_dir = _mk_local_tmp("bipia-prepare-fail")
    bench = tmp_dir / "benchmark"
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        out = bench / rel
        if path.is_dir():
            out.mkdir(parents=True, exist_ok=True)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(path.read_bytes())
    (bench / "qa" / "train.jsonl").unlink()
    (bench / "qa" / "test.jsonl").unlink()

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_bipia_contexts.py",
            "--benchmark-root",
            str(bench.as_posix()),
            "--strict",
            "--artifacts-root",
            str((tmp_dir / "artifacts").as_posix()),
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 1
    payload = json.loads(proc.stdout)
    report = json.loads(Path(payload["report"]).read_text(encoding="utf-8"))
    assert report["status"] == "FAILED"
    assert any("newsqa-data-dir" in f for f in report["failures"])
