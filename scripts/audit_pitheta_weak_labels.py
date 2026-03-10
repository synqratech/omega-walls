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
from omega.eval.pitheta_weak_label_audit import evaluate_weak_label_audit, write_weak_label_audit_artifacts
from omega.pitheta.dataset_builder import load_pitheta_jsonl
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2


def _parse_qualities(raw: str) -> list[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Mini-audit for PiTheta weak labels on train split.")
    parser.add_argument("--data-dir", required=True, help="Path containing train.jsonl")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--sample-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--include-qualities", default="weak,silver")
    parser.add_argument(
        "--semantic-mode",
        choices=["off", "on", "current"],
        default="off",
        help="Projector semantic mode for audit re-projection.",
    )
    parser.add_argument("--artifacts-root", default="artifacts/pitheta_weak_label_audit")
    args = parser.parse_args()

    if int(args.sample_size) < 1:
        raise ValueError("sample-size must be >= 1")
    if int(args.sample_size) > 5000:
        raise ValueError("sample-size is too large for mini-audit")

    overrides = None
    if args.semantic_mode == "off":
        overrides = {"pi0": {"semantic": {"enabled": "false"}}}
    elif args.semantic_mode == "on":
        overrides = {"pi0": {"semantic": {"enabled": "true"}}}

    snapshot = load_resolved_config(profile=str(args.profile), cli_overrides=overrides)
    cfg = snapshot.resolved
    projector = Pi0IntentAwareV2(cfg)

    train_path = Path(args.data_dir) / "train.jsonl"
    rows = load_pitheta_jsonl(train_path.as_posix())
    qualities = _parse_qualities(args.include_qualities)

    summary, review_rows = evaluate_weak_label_audit(
        projector=projector,
        rows=rows,
        sample_size=int(args.sample_size),
        seed=int(args.seed),
        include_qualities=qualities,
    )
    if summary["sampled_count"] == 0:
        print(
            json.dumps(
                {
                    "status": "NO_GO",
                    "reason": "no rows matched include_qualities",
                    "include_qualities": qualities,
                },
                ensure_ascii=True,
                indent=2,
            )
        )
        return 1

    run_id = f"pitheta_weak_label_audit_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{snapshot.resolved_sha256[:12]}"
    out_dir = ROOT / str(args.artifacts_root) / run_id
    report = {
        "status": "ok",
        "run_id": run_id,
        "profile": str(args.profile),
        "resolved_config_sha256": snapshot.resolved_sha256,
        "data_dir": str(Path(args.data_dir).as_posix()),
        "train_path": str(train_path.as_posix()),
        "semantic_mode": str(args.semantic_mode),
        "include_qualities": qualities,
        **summary,
    }
    artifacts = write_weak_label_audit_artifacts(
        out_dir=out_dir.as_posix(),
        report=report,
        review_rows=review_rows,
    )
    report["artifacts"] = artifacts
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

