from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIZES = ("short", "medium", "large")
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_ERROR = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _extract_json_blob(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("stdout does not contain JSON payload")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected object JSON: {path}")
    return payload


def _require_baseline_shape(baseline: Dict[str, Any]) -> None:
    required_root = ("schema_version", "python", "profile", "metric", "benchmark_args", "p95_ms")
    for key in required_root:
        if key not in baseline:
            raise ValueError(f"baseline missing required key: {key}")
    p95 = baseline.get("p95_ms")
    if not isinstance(p95, dict):
        raise ValueError("baseline.p95_ms must be an object")
    for size in SIZES:
        if size not in p95:
            raise ValueError(f"baseline.p95_ms missing size: {size}")
        value = float(p95[size])
        if value <= 0.0:
            raise ValueError(f"baseline.p95_ms.{size} must be > 0")


def _extract_candidate_p95(report: Dict[str, Any]) -> Dict[str, float]:
    modes = report.get("modes")
    if not isinstance(modes, dict):
        raise ValueError("benchmark report missing modes")
    pi0 = modes.get("omega_rule_only_pi0")
    if not isinstance(pi0, dict):
        raise ValueError("benchmark report missing modes.omega_rule_only_pi0")
    per_size = pi0.get("per_size")
    if not isinstance(per_size, dict):
        raise ValueError("benchmark report missing omega_rule_only_pi0.per_size")

    out: Dict[str, float] = {}
    for size in SIZES:
        size_obj = per_size.get(size)
        if not isinstance(size_obj, dict):
            raise ValueError(f"benchmark report missing per_size.{size}")
        p95_value = float(size_obj.get("p95_ms"))
        if p95_value <= 0.0:
            raise ValueError(f"candidate p95 must be > 0 for {size}")
        out[size] = p95_value
    return out


def _build_benchmark_cmd(
    *,
    python_exec: str,
    profile: str,
    artifacts_root: str,
    run_tag: str,
    benchmark_args: Dict[str, Any],
) -> List[str]:
    cmd = [
        python_exec,
        str((ROOT / "scripts" / "benchmark_omega_latency.py").resolve()),
        "--profile",
        profile,
        "--artifacts-root",
        artifacts_root,
        "--run-tag",
        run_tag,
    ]
    if "short_chars" in benchmark_args:
        cmd.extend(["--short-chars", str(int(benchmark_args["short_chars"]))])
    if "medium_chars" in benchmark_args:
        cmd.extend(["--medium-chars", str(int(benchmark_args["medium_chars"]))])
    if "large_chars" in benchmark_args:
        cmd.extend(["--large-chars", str(int(benchmark_args["large_chars"]))])
    if "repeats" in benchmark_args:
        cmd.extend(["--repeats", str(int(benchmark_args["repeats"]))])
    return cmd


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FW-001 perf gate checker (pi0 p95 overhead vs frozen baseline).")
    parser.add_argument(
        "--baseline-file",
        default="config/perf_baselines/fw001_pi0_py313_ubuntu.json",
        help="Frozen baseline JSON path.",
    )
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--artifacts-root", default="artifacts/fw001_perf")
    parser.add_argument("--perf-overhead-max", type=float, default=0.15)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args(argv)

    try:
        baseline_path = (ROOT / str(args.baseline_file)).resolve()
        baseline = _load_json(baseline_path)
        _require_baseline_shape(baseline)

        benchmark_args = baseline.get("benchmark_args", {})
        if not isinstance(benchmark_args, dict):
            raise ValueError("baseline.benchmark_args must be an object")

        run_tag = str(args.run_tag or f"fw001_perf_gate_{int(time.time())}")
        cmd = _build_benchmark_cmd(
            python_exec=sys.executable,
            profile=str(args.profile),
            artifacts_root=str(args.artifacts_root),
            run_tag=run_tag,
            benchmark_args=benchmark_args,
        )
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
        if proc.returncode != 0:
            raise RuntimeError(f"benchmark_omega_latency.py failed with exit code {proc.returncode}")
        launcher_payload = _extract_json_blob(proc.stdout)
        report_path = Path(str(launcher_payload.get("report_json", ""))).resolve()
        if not report_path.exists():
            raise FileNotFoundError(f"benchmark report not found: {report_path}")

        candidate_report = _load_json(report_path)
        candidate_p95 = _extract_candidate_p95(candidate_report)
        baseline_p95 = {size: float(baseline["p95_ms"][size]) for size in SIZES}

        threshold = float(args.perf_overhead_max)
        comparisons: Dict[str, Any] = {}
        failures: List[str] = []
        for size in SIZES:
            base = baseline_p95[size]
            cand = candidate_p95[size]
            overhead = (cand - base) / base
            passed = overhead <= threshold
            comparisons[size] = {
                "baseline_p95_ms": base,
                "candidate_p95_ms": cand,
                "overhead_ratio": overhead,
                "threshold": threshold,
                "status": "PASS" if passed else "FAIL",
            }
            if not passed:
                failures.append(size)

        overall_pass = not failures
        status = "PASS" if overall_pass else "FAIL"
        out_dir = report_path.parent
        gate_report_path = out_dir / "gate_report.json"
        gate_report = {
            "schema_version": "fw001_perf_gate_report_v1",
            "created_at_utc": _utc_now(),
            "status": status,
            "strict": bool(args.strict),
            "threshold": threshold,
            "baseline_file": str(baseline_path),
            "benchmark_report": str(report_path),
            "comparisons": comparisons,
            "failed_sizes": failures,
        }
        gate_report_path.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        print(
            json.dumps(
                {
                    "status": status.lower(),
                    "gate_report": str(gate_report_path),
                    "failed_sizes": failures,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        if overall_pass:
            return EXIT_PASS
        return EXIT_FAIL if args.strict else EXIT_PASS
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False, indent=2))
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
