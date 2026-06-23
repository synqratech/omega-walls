from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent

MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")

BENCH_ROOT = ROOT / "benchmarks"
RESULTS_ROOT = BENCH_ROOT / "results"

TRACKS = ["agentdojo", "agent3sigma", "wainjectbench", "promptshield"]
TRACK_FILES = ["README.md", "DOWNLOAD.md", "RUN.md", "EXPECTED_ARTIFACTS.md"]

SNAPSHOT_FILES = {
    "agentdojo": "agentdojo_frozen_20260330.json",
    "agent3sigma_stage": "agent3sigma_frozen_20260622.json",
    "wainjectbench_text": "wainject_frozen_20260324.json",
    "promptshield_text": "promptshield_frozen_20260324.json",
}

SMOKE_COMMANDS = [
    [sys.executable, "scripts/eval_agentdojo_stateful_mini.py", "--help"],
    [sys.executable, "scripts/eval_wainjectbench_text.py", "--help"],
    [sys.executable, "scripts/eval_promptshield_text.py", "--help"],
]


def _iter_markdown_links(text: str) -> Iterable[str]:
    for m in MD_LINK_RE.finditer(text):
        yield str(m.group(1)).strip()


def _assert_exists(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"missing required path: {path}")


def _check_english_only_markdown(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if CYRILLIC_RE.search(text):
        raise RuntimeError(f"non-English (Cyrillic) content detected: {path}")


def _check_markdown_links(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for link in _iter_markdown_links(text):
        if not link or link.startswith("#"):
            continue
        low = link.lower()
        if low.startswith(("http://", "https://", "mailto:")):
            continue
        target = (path.parent / link.split("#", 1)[0]).resolve()
        if not target.exists():
            raise RuntimeError(f"broken local link in {path}: {link}")


def _check_benchmark_structure() -> None:
    _assert_exists(BENCH_ROOT / "README.md")
    for track in TRACKS:
        track_dir = BENCH_ROOT / track
        _assert_exists(track_dir)
        for rel in TRACK_FILES:
            _assert_exists(track_dir / rel)


def _check_markdown_contract() -> None:
    for md in BENCH_ROOT.rglob("*.md"):
        _check_english_only_markdown(md)
        _check_markdown_links(md)


def _check_results_contract(*, require_source_reports: bool) -> None:
    _assert_exists(RESULTS_ROOT / "benchmark_index.json")
    _assert_exists(RESULTS_ROOT / "frozen_scorecard_top3.csv")

    index = json.loads((RESULTS_ROOT / "benchmark_index.json").read_text(encoding="utf-8"))
    if str(index.get("schema_version", "")) != "benchmark_public_index_v1":
        raise RuntimeError("benchmark_index.json has invalid schema_version")

    rows = index.get("benchmarks", [])
    if not isinstance(rows, list) or len(rows) != 4:
        raise RuntimeError("benchmark_index.json must contain exactly 4 benchmark entries")

    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("invalid benchmark entry type in benchmark_index.json")
        benchmark_id = str(row.get("benchmark_id", "")).strip()
        frozen_run_id = str(row.get("frozen_run_id", "")).strip()
        snapshot_json = str(row.get("snapshot_json", "")).strip()
        status = str(row.get("comparability_status", "")).strip()
        if not benchmark_id or not frozen_run_id or not snapshot_json or not status:
            raise RuntimeError(f"incomplete benchmark index row: {row}")
        seen.add(benchmark_id)

        snapshot_path = ROOT / snapshot_json
        _assert_exists(snapshot_path)
        snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if str(snap.get("schema_version", "")) != "benchmark_snapshot_v1":
            raise RuntimeError(f"invalid snapshot schema version: {snapshot_path}")
        if str(snap.get("frozen_run_id", "")) != frozen_run_id:
            raise RuntimeError(f"run_id mismatch between index and snapshot: {snapshot_path}")
        if str(snap.get("status", "")) != "ok":
            raise RuntimeError(f"snapshot status is not ok: {snapshot_path}")
        if "metrics" not in snap:
            raise RuntimeError(f"snapshot missing metrics: {snapshot_path}")

        source_report = str(snap.get("source_report", "")).strip()
        if not source_report:
            raise RuntimeError(f"snapshot missing source_report: {snapshot_path}")
        if require_source_reports:
            report_path = ROOT / source_report
            _assert_exists(report_path)

    required_ids = set(SNAPSHOT_FILES.keys())
    if seen != required_ids:
        raise RuntimeError(f"benchmark ids mismatch: expected={sorted(required_ids)} actual={sorted(seen)}")

    with (RESULTS_ROOT / "frozen_scorecard_top3.csv").open("r", encoding="utf-8") as fh:
        parsed = list(csv.DictReader(fh))
    if len(parsed) < 9:
        raise RuntimeError("frozen_scorecard_top3.csv has too few rows")
    for row in parsed:
        for col in ("benchmark_id", "frozen_run_id", "metric", "value", "comparability_status"):
            if not str(row.get(col, "")).strip():
                raise RuntimeError(f"csv row missing required field '{col}': {row}")


def _run_smoke_commands() -> None:
    for cmd in SMOKE_COMMANDS:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                "benchmark smoke command failed: "
                + " ".join(cmd)
                + f"\nexit={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
            )


def validate(*, smoke_exec: bool, require_source_reports: bool) -> dict:
    _check_benchmark_structure()
    _check_markdown_contract()
    _check_results_contract(require_source_reports=require_source_reports)
    if smoke_exec:
        _run_smoke_commands()

    return {
        "event": "benchmarks_contract_v1",
        "status": "ok",
        "smoke_exec": bool(smoke_exec),
        "require_source_reports": bool(require_source_reports),
        "tracks": TRACKS,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate public benchmarks contract and snapshots.")
    parser.add_argument("--smoke-exec", action="store_true", help="Run command-surface smoke checks (--help).")
    parser.add_argument(
        "--require-source-reports",
        action="store_true",
        help="Require source report paths from snapshots to exist in local repository.",
    )
    args = parser.parse_args()

    payload = validate(smoke_exec=bool(args.smoke_exec), require_source_reports=bool(args.require_source_reports))
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
