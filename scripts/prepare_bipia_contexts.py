from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.eval.bipia_manifest import verify_qa_abstract_md5


def _pick_python() -> str:
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        try:
            probe = subprocess.run(
                [str(venv_python), "--version"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if int(probe.returncode) == 0:
                return str(venv_python)
        except Exception:
            pass
    return sys.executable


def _run_command(argv: List[str], cwd: Path, stdin_text: str | None = None) -> Dict[str, Any]:
    start = time.time()
    proc = subprocess.run(
        argv,
        cwd=str(cwd),
        input=stdin_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return {
        "argv": argv,
        "cwd": str(cwd.as_posix()),
        "exit_code": int(proc.returncode),
        "duration_sec": round(time.time() - start, 3),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _expected_context_paths(benchmark_root: Path) -> Dict[str, List[Path]]:
    return {
        "qa": [benchmark_root / "qa" / "train.jsonl", benchmark_root / "qa" / "test.jsonl"],
        "abstract": [benchmark_root / "abstract" / "train.jsonl", benchmark_root / "abstract" / "test.jsonl"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and validate BIPIA qa/abstract contexts")
    parser.add_argument("--benchmark-root", default="data/BIPIA-main/benchmark")
    parser.add_argument("--newsqa-data-dir", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--artifacts-root", default="artifacts/bipia_prepare")
    args = parser.parse_args()

    benchmark_root = (ROOT / args.benchmark_root) if not Path(args.benchmark_root).is_absolute() else Path(args.benchmark_root)
    run_id = datetime.now(timezone.utc).strftime("prepare_%Y%m%dT%H%M%SZ")
    out_dir = ROOT / args.artifacts_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = _expected_context_paths(benchmark_root)
    existing_before = {
        task: {path.name: path.exists() for path in paths}
        for task, paths in expected.items()
    }

    commands: Dict[str, Dict[str, Any]] = {}
    failures: List[str] = []
    python_exec = _pick_python()

    qa_ready_before = all(path.exists() for path in expected["qa"])
    if not qa_ready_before:
        if not args.newsqa_data_dir:
            failures.append("qa contexts missing and --newsqa-data-dir not provided")
        else:
            commands["qa_process"] = _run_command(
                [
                    python_exec,
                    "process.py",
                    "--data_dir",
                    str(args.newsqa_data_dir),
                ],
                cwd=benchmark_root / "qa",
            )
            if commands["qa_process"]["exit_code"] != 0:
                failures.append("qa process.py failed")

    abstract_ready_before = all(path.exists() for path in expected["abstract"])
    if not abstract_ready_before:
        commands["abstract_process"] = _run_command(
            [python_exec, "process.py"],
            cwd=benchmark_root / "abstract",
            stdin_text="Y\n",
        )
        if commands["abstract_process"]["exit_code"] != 0:
            failures.append("abstract process.py failed")

    qa_abstract_md5_ok, md5_checks = verify_qa_abstract_md5(str(benchmark_root))
    if not qa_abstract_md5_ok:
        failures.append("qa/abstract md5 verification failed")

    existing_after = {
        task: {path.name: path.exists() for path in paths}
        for task, paths in expected.items()
    }

    report = {
        "run_id": run_id,
        "benchmark_root": str(benchmark_root.as_posix()),
        "strict": bool(args.strict),
        "newsqa_data_dir": args.newsqa_data_dir,
        "existing_before": existing_before,
        "existing_after": existing_after,
        "qa_abstract_md5_ok": qa_abstract_md5_ok,
        "md5_checks": md5_checks,
        "commands": commands,
        "failures": failures,
        "status": "OK" if not failures else "FAILED",
    }
    report_path = out_dir / "prepare_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    print(json.dumps({"run_id": run_id, "status": report["status"], "report": str(report_path)}, ensure_ascii=True, indent=2))
    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
