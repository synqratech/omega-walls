from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import config_refs_from_snapshot, load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.eval.deepset_manifest import DeepsetManifestInput, build_deepset_manifest, write_manifest
from omega.eval.deepset_metrics import evaluate_deepset, evaluate_deepset_thresholds
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector


def _parse_seed_pack(raw: str | None, repeats: int, seed_start: int) -> List[int]:
    if raw:
        out = [int(x.strip()) for x in raw.split(",") if x.strip()]
        if not out:
            raise ValueError("seed-pack is empty")
        return out
    return [seed_start + i for i in range(repeats)]


def _stddev(values: List[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.pstdev(values))


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Deepset Validation Report",
        "",
        f"- run_id: `{report['run_id']}`",
        f"- status: **{report['status']}**",
        f"- mode: `{report['mode']}`",
        f"- split: `{report['split']}`",
        f"- repeats: `{report['repeats']}`",
        f"- seeds: `{report['seed_pack']}`",
        "",
        "## Aggregate",
        f"- attack_off_rate_mean: `{report['aggregate']['attack_off_rate_mean']:.6f}`",
        f"- attack_off_rate_stddev: `{report['aggregate']['attack_off_rate_stddev']:.6f}`",
        f"- benign_off_rate_mean: `{report['aggregate']['benign_off_rate_mean']:.6f}`",
        f"- benign_off_rate_stddev: `{report['aggregate']['benign_off_rate_stddev']:.6f}`",
        f"- f1_mean: `{report['aggregate']['f1_mean']:.6f}`",
        f"- f1_stddev: `{report['aggregate']['f1_stddev']:.6f}`",
        f"- coverage_wall_any_attack_mean: `{report['aggregate']['coverage_wall_any_attack_mean']:.6f}`",
        "",
    ]
    if report["failures"]:
        lines.append("## Failures")
        for failure in report["failures"]:
            lines.append(f"- {failure}")
    else:
        lines.append("All checks passed.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full deepset validation and stability report")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--benchmark-root", default=None)
    parser.add_argument("--mode", choices=["sampled", "full"], default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed-pack", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--artifacts-root", default="artifacts/deepset_validation")
    parser.add_argument("--require-semantic", action="store_true")
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved
    deepset_cfg = cfg.get("deepset", {})

    mode = str(args.mode or deepset_cfg.get("mode_default", "full")).lower()
    split = str(args.split or deepset_cfg.get("split_default", "test")).lower()
    benchmark_root = str(args.benchmark_root or deepset_cfg.get("benchmark_root", "data/deepset-prompt-injections"))
    sampled_cfg = deepset_cfg.get("sampled", {}) or {}
    max_samples = int(args.max_samples or sampled_cfg.get("max_samples", 116))
    thresholds_cfg = (deepset_cfg.get("thresholds", {}) or {}).get("report", {}) or {}
    label_attack_value = int(deepset_cfg.get("label_attack_value", 1))
    repro_cfg = deepset_cfg.get("reproducibility", {}) or {}
    seed_start = int(args.seed_start if args.seed_start is not None else repro_cfg.get("seed_default", 41))
    seed_pack = _parse_seed_pack(args.seed_pack, repeats=args.repeats, seed_start=seed_start)
    if args.repeats != len(seed_pack):
        args.repeats = len(seed_pack)

    run_id = f"deepset_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{snapshot.resolved_sha256[:12]}"
    run_dir = ROOT / args.artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_deepset_manifest(
        DeepsetManifestInput(
            benchmark_root=benchmark_root,
            split=split,
            mode=mode,
            seed_pack=seed_pack,
            label_attack_value=label_attack_value,
            config_refs=config_refs_from_snapshot(snapshot, code_commit="local"),
            strict=True,
        )
    )
    manifest_path = run_dir / "deepset_manifest.json"
    write_manifest(str(manifest_path), manifest)

    projector = build_projector(cfg)
    if args.require_semantic and not projector.ensure_semantic_active():
        status = projector.semantic_status()
        print(json.dumps({"status": "NO_GO", "reason": status.get("error", "semantic inactive")}, ensure_ascii=True))
        return 1
    omega_core = OmegaCoreV1(omega_params_from_config(cfg))
    off_policy = OffPolicyV1(cfg)

    failures: List[str] = []
    run_reports: List[Dict[str, Any]] = []
    for idx, seed in enumerate(seed_pack, start=1):
        report = evaluate_deepset(
            projector=projector,
            omega_core=omega_core,
            off_policy=off_policy,
            benchmark_root=benchmark_root,
            split=split,
            mode=mode,
            max_samples=max_samples,
            seed=seed,
            label_attack_value=label_attack_value,
        )
        report["manifest_path"] = str(manifest_path.relative_to(ROOT).as_posix())
        report["data_readiness"] = manifest.get("data_readiness", {})
        run_failures = evaluate_deepset_thresholds(report, thresholds=thresholds_cfg if isinstance(thresholds_cfg, dict) else {})
        if run_failures:
            report["status"] = "fail"
            report["threshold_failures"] = run_failures
            for entry in run_failures:
                failures.append(f"run_{idx}: {entry}")
        run_path = run_dir / f"run_{idx}.json"
        run_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
        run_reports.append(report)

    attack_rates = [float(r["metrics"]["attack_off_rate"]) for r in run_reports]
    benign_rates = [float(r["metrics"]["benign_off_rate"]) for r in run_reports]
    f1_values = [float(r["metrics"]["f1"]) for r in run_reports]
    coverage_values = [float(r["metrics"]["coverage_wall_any_attack"]) for r in run_reports]

    aggregate = {
        "attack_off_rate_mean": float(statistics.fmean(attack_rates)) if attack_rates else 0.0,
        "attack_off_rate_stddev": _stddev(attack_rates),
        "benign_off_rate_mean": float(statistics.fmean(benign_rates)) if benign_rates else 0.0,
        "benign_off_rate_stddev": _stddev(benign_rates),
        "f1_mean": float(statistics.fmean(f1_values)) if f1_values else 0.0,
        "f1_stddev": _stddev(f1_values),
        "coverage_wall_any_attack_mean": float(statistics.fmean(coverage_values)) if coverage_values else 0.0,
        "coverage_wall_any_attack_stddev": _stddev(coverage_values),
    }

    summary = {
        "run_id": run_id,
        "status": "GO" if not failures else "NO_GO",
        "profile": args.profile,
        "mode": mode,
        "split": split,
        "repeats": args.repeats,
        "seed_pack": seed_pack,
        "manifest_path": str(manifest_path.relative_to(ROOT).as_posix()),
        "runs": [f"run_{i}.json" for i in range(1, len(run_reports) + 1)],
        "aggregate": aggregate,
        "failures": failures,
    }

    stability_json = run_dir / "stability_report.json"
    stability_md = run_dir / "stability_report.md"
    stability_json.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")
    stability_md.write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))

    if args.strict and failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
