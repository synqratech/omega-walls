from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.core.omega_core import OmegaCoreV1
from omega.core.params import omega_params_from_config
from omega.eval.deepset_fn_analysis import evaluate_deepset_records, summarize_fn_patterns, write_fn_artifacts
from omega.policy.off_policy_v1 import OffPolicyV1
from omega.projector.factory import build_projector


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze false negatives on deepset benchmark")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--benchmark-root", default="data/deepset-prompt-injections")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--mode", choices=["sampled", "full"], default="full")
    parser.add_argument("--max-samples", type=int, default=116)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--label-attack-value", type=int, default=1)
    parser.add_argument("--require-semantic", action="store_true")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--artifacts-root", default="artifacts/deepset_analysis")
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=str(args.profile))
    cfg = snapshot.resolved
    projector = build_projector(cfg)
    if args.require_semantic and not projector.ensure_semantic_active():
        status = projector.semantic_status()
        print(json.dumps({"status": "NO_GO", "reason": status.get("error", "semantic inactive")}, ensure_ascii=True))
        return 1

    omega_core = OmegaCoreV1(omega_params_from_config(cfg))
    off_policy = OffPolicyV1(cfg)

    records = evaluate_deepset_records(
        projector=projector,
        omega_core=omega_core,
        off_policy=off_policy,
        benchmark_root=str(args.benchmark_root),
        split=str(args.split),
        mode=str(args.mode),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
        label_attack_value=int(args.label_attack_value),
    )
    summary = summarize_fn_patterns(records, top_k=int(args.top_k))
    run_id = f"deepset_fn_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{snapshot.resolved_sha256[:12]}"
    out_dir = ROOT / str(args.artifacts_root) / run_id
    artifacts = write_fn_artifacts(out_dir=out_dir.as_posix(), records=records, summary=summary)

    report = {
        "run_id": run_id,
        "status": "ok",
        "profile": str(args.profile),
        "split": str(args.split),
        "mode": str(args.mode),
        "seed": int(args.seed),
        "resolved_config_sha256": snapshot.resolved_sha256,
        "summary": summary,
        "artifacts": artifacts,
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
