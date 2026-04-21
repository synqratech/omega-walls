from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
import time
import uuid
from typing import Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from omega import OmegaWalls


@dataclass
class TimingStats:
    count: int
    first_ms: float
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _stats_ms(samples_ms: List[float]) -> TimingStats:
    return TimingStats(
        count=len(samples_ms),
        first_ms=float(samples_ms[0] if samples_ms else 0.0),
        mean_ms=float(statistics.fmean(samples_ms) if samples_ms else 0.0),
        median_ms=float(statistics.median(samples_ms) if samples_ms else 0.0),
        p95_ms=float(_percentile(samples_ms, 0.95)),
        min_ms=float(min(samples_ms) if samples_ms else 0.0),
        max_ms=float(max(samples_ms) if samples_ms else 0.0),
        std_ms=float(statistics.pstdev(samples_ms) if len(samples_ms) > 1 else 0.0),
    )


def _make_text(source: str, target_chars: int) -> str:
    if target_chars <= 0:
        return ""
    base = " ".join((source or "").split())
    if not base:
        base = "Operational security memo for prompt-injection robustness testing."
    chunks = [base]
    while sum(len(x) for x in chunks) < target_chars:
        chunks.append(base)
    joined = "\n".join(chunks)
    return joined[:target_chars]


def _load_source_text() -> str:
    candidates = [
        ROOT / "docs" / "architecture.md",
        ROOT / "README.md",
    ]
    for path in candidates:
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return "Omega Walls latency benchmark source text."


def _measure_loop(fn: Callable[[int], None], repeats: int) -> TimingStats:
    samples_ms: List[float] = []
    for idx in range(repeats):
        t0 = time.perf_counter()
        fn(idx)
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
    return _stats_ms(samples_ms)


def _build_rule_guard(profile: str) -> Dict[str, object]:
    t0 = time.perf_counter()
    guard = OmegaWalls(profile=profile, projector_mode="pi0")
    t1 = time.perf_counter()
    return {"guard": guard, "init_ms": (t1 - t0) * 1000.0}


def _build_hybrid_guard(
    profile: str,
    api_model: str,
    cache_path: Path,
    error_log_path: Path,
) -> Dict[str, object]:
    t0 = time.perf_counter()
    guard = OmegaWalls(
        profile=profile,
        projector_mode="hybrid_api",
        api_model=api_model,
        cli_overrides={
            "projector": {
                "fallback_to_pi0": False,
                "api_perception": {
                    "enabled": "true",
                    "strict": True,
                    "cache_path": str(cache_path),
                    "error_log_path": str(error_log_path),
                },
            }
        },
    )
    t1 = time.perf_counter()
    return {"guard": guard, "init_ms": (t1 - t0) * 1000.0}


def _format_row(name: str, stats: TimingStats) -> str:
    return (
        f"{name:26} "
        f"mean={stats.mean_ms:8.1f} ms  "
        f"p50={stats.median_ms:8.1f} ms  "
        f"p95={stats.p95_ms:8.1f} ms  "
        f"first={stats.first_ms:8.1f} ms"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Latency benchmark for OmegaWalls (baseline/rule/hybrid_api).")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--api-model", default="gpt-5.4-mini")
    parser.add_argument("--short-chars", type=int, default=280)
    parser.add_argument("--medium-chars", type=int, default=4200)
    parser.add_argument("--large-chars", type=int, default=12000)
    parser.add_argument("--repeats", type=int, default=4, help="Fallback repeats for baseline/rule-only.")
    parser.add_argument("--repeats-short", type=int, default=None)
    parser.add_argument("--repeats-medium", type=int, default=None)
    parser.add_argument("--repeats-large", type=int, default=None)
    parser.add_argument("--hybrid-repeats", type=int, default=2, help="Fallback repeats for hybrid_api cold/warm.")
    parser.add_argument("--hybrid-repeats-short", type=int, default=None)
    parser.add_argument("--hybrid-repeats-medium", type=int, default=None)
    parser.add_argument("--hybrid-repeats-large", type=int, default=None)
    parser.add_argument("--artifacts-root", default="artifacts/latency_bench")
    parser.add_argument("--run-tag", default=None)
    args = parser.parse_args()

    run_id = str(args.run_tag or f"omega_latency_{_utc_now()}")
    out_dir = (ROOT / str(args.artifacts_root) / run_id).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_text = _load_source_text()
    inputs = {
        "short": _make_text(source_text, int(args.short_chars)),
        "medium": _make_text(source_text, int(args.medium_chars)),
        "large": _make_text(source_text, int(args.large_chars)),
    }

    repeats_by_size = {
        "short": int(args.repeats_short if args.repeats_short is not None else args.repeats),
        "medium": int(args.repeats_medium if args.repeats_medium is not None else args.repeats),
        "large": int(args.repeats_large if args.repeats_large is not None else args.repeats),
    }
    hybrid_repeats_by_size = {
        "short": int(args.hybrid_repeats_short if args.hybrid_repeats_short is not None else args.hybrid_repeats),
        "medium": int(args.hybrid_repeats_medium if args.hybrid_repeats_medium is not None else args.hybrid_repeats),
        "large": int(args.hybrid_repeats_large if args.hybrid_repeats_large is not None else args.hybrid_repeats),
    }

    report: Dict[str, object] = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": str(args.profile),
        "api_model": str(args.api_model),
        "inputs_chars": {k: len(v) for k, v in inputs.items()},
        "repeats_by_size": repeats_by_size,
        "hybrid_repeats_by_size": hybrid_repeats_by_size,
        "modes": {},
    }

    # Baseline (no Omega): trivial pass-through function.
    baseline_results: Dict[str, Dict[str, float]] = {}
    for key, text in inputs.items():
        def _baseline(_: int, payload: str = text) -> None:
            _ = len(payload)

        stats = _measure_loop(_baseline, int(repeats_by_size[key]))
        baseline_results[key] = asdict(stats)
    report["modes"]["baseline_no_omega"] = {
        "init_ms": 0.0,
        "per_size": baseline_results,
    }

    # Rule-only (pi0).
    rule_built = _build_rule_guard(profile=str(args.profile))
    rule_guard: OmegaWalls = rule_built["guard"]  # type: ignore[assignment]
    rule_results: Dict[str, Dict[str, float]] = {}
    for key, text in inputs.items():
        def _rule(_: int, payload: str = text, size_key: str = key) -> None:
            _ = rule_guard.analyze_text(
                payload,
                session_id=f"lat-rule-{size_key}",
                source_id=f"latency:rule:{size_key}",
                source_type="doc",
                trust="untrusted",
                reset_session=True,
            )

        stats = _measure_loop(_rule, int(repeats_by_size[key]))
        rule_results[key] = asdict(stats)
    report["modes"]["omega_rule_only_pi0"] = {
        "init_ms": float(rule_built["init_ms"]),
        "per_size": rule_results,
    }

    # Hybrid API mode.
    if not str(os.getenv("OPENAI_API_KEY", "")).strip():
        report["modes"]["omega_hybrid_api"] = {"status": "skipped", "reason": "missing_env:OPENAI_API_KEY"}
    else:
        cache_path = out_dir / "hybrid_api_cache.jsonl"
        error_log = out_dir / "hybrid_api_errors.jsonl"
        hybrid_built = _build_hybrid_guard(
            profile=str(args.profile),
            api_model=str(args.api_model),
            cache_path=cache_path,
            error_log_path=error_log,
        )
        hybrid_guard: OmegaWalls = hybrid_built["guard"]  # type: ignore[assignment]

        hybrid_cold: Dict[str, Dict[str, float]] = {}
        hybrid_warm: Dict[str, Dict[str, float]] = {}

        for key, text in inputs.items():
            stable_text = text

            def _hybrid_warm(_: int, payload: str = stable_text, size_key: str = key) -> None:
                _ = hybrid_guard.analyze_text(
                    payload,
                    session_id=f"lat-hybrid-warm-{size_key}",
                    source_id=f"latency:hybrid:warm:{size_key}",
                    source_type="doc",
                    trust="untrusted",
                    reset_session=True,
                )

            def _hybrid_cold(i: int, payload: str = stable_text, size_key: str = key) -> None:
                nonce = uuid.uuid4().hex[:12]
                variant = f"{payload}\n\nnonce:{size_key}:{i}:{nonce}"
                _ = hybrid_guard.analyze_text(
                    variant,
                    session_id=f"lat-hybrid-cold-{size_key}",
                    source_id=f"latency:hybrid:cold:{size_key}",
                    source_type="doc",
                    trust="untrusted",
                    reset_session=True,
                )

            warm_stats = _measure_loop(_hybrid_warm, int(hybrid_repeats_by_size[key]))
            cold_stats = _measure_loop(_hybrid_cold, int(hybrid_repeats_by_size[key]))
            hybrid_warm[key] = asdict(warm_stats)
            hybrid_cold[key] = asdict(cold_stats)

        report["modes"]["omega_hybrid_api"] = {
            "status": "ok",
            "init_ms": float(hybrid_built["init_ms"]),
            "cache_path": str(cache_path),
            "error_log_path": str(error_log),
            "per_size_warm": hybrid_warm,
            "per_size_cold": hybrid_cold,
        }

    out_json = out_dir / "report.json"
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"status": "ok", "run_id": run_id, "report_json": str(out_json)}, ensure_ascii=False, indent=2))
    print("")
    print("=== Latency Summary (ms) ===")
    for size_key in ("short", "medium", "large"):
        print(f"\n[{size_key}] chars={len(inputs[size_key])}")
        b = TimingStats(**report["modes"]["baseline_no_omega"]["per_size"][size_key])  # type: ignore[index]
        r = TimingStats(**report["modes"]["omega_rule_only_pi0"]["per_size"][size_key])  # type: ignore[index]
        print(_format_row("baseline_no_omega", b))
        print(_format_row("omega_rule_only_pi0", r))
        hybrid = report["modes"].get("omega_hybrid_api", {})
        if isinstance(hybrid, dict) and hybrid.get("status") == "ok":
            w = TimingStats(**hybrid["per_size_warm"][size_key])  # type: ignore[index]
            c = TimingStats(**hybrid["per_size_cold"][size_key])  # type: ignore[index]
            print(_format_row("omega_hybrid_api_warm", w))
            print(_format_row("omega_hybrid_api_cold", c))
        else:
            print("omega_hybrid_api           skipped (missing OPENAI_API_KEY)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
