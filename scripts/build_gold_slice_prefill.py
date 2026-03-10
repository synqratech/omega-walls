from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.pitheta.gold_slice_prefill import (
    PrefillConfig,
    build_gold_slice_prefill,
    write_json,
    write_jsonl,
)
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2


def _parse_weights(raw: str, keys: list[str]) -> Dict[str, float]:
    values = [x.strip() for x in str(raw).split(",") if x.strip()]
    if len(values) != len(keys):
        raise ValueError(f"expected {len(keys)} comma-separated values in '{raw}'")
    out: Dict[str, float] = {}
    for idx, key in enumerate(keys):
        out[key] = float(values[idx])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build gold-slice prefill candidates (source/chunk stratified).")
    parser.add_argument("--target-size", type=int, default=220)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--deepset-root", default="data/deepset-prompt-injections")
    parser.add_argument("--wainject-root", default="data/WAInjectBench/text")
    parser.add_argument("--source-weights", default="0.45,0.35,0.20", help="deepset,wainject,redteam")
    parser.add_argument("--chunk-weights", default="0.30,0.50,0.20", help="64,128_256,512")
    parser.add_argument("--ordinal-bins", default="0.45,1.10,2.00")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--write-annotator-seeds", action="store_true")
    parser.add_argument("--semantic-labeling", action="store_true", help="Enable pi0 semantic layer for weak-label prefill.")
    args = parser.parse_args()

    if int(args.target_size) < 1:
        raise ValueError("target-size must be >= 1")

    source_weights = _parse_weights(str(args.source_weights), ["deepset_train", "wainject_text", "redteam_synth_train"])
    chunk_weights = _parse_weights(str(args.chunk_weights), ["64", "128_256", "512"])
    ordinal_bins_vals = [float(x.strip()) for x in str(args.ordinal_bins).split(",") if x.strip()]
    if len(ordinal_bins_vals) != 3:
        raise ValueError("ordinal-bins must contain 3 comma-separated values")

    overrides = None
    if not bool(args.semantic_labeling):
        overrides = {"pi0": {"semantic": {"enabled": "false"}}}
    snapshot = load_resolved_config(profile=str(args.profile), cli_overrides=overrides)
    projector = Pi0IntentAwareV2(snapshot.resolved)
    cfg = PrefillConfig(
        target_size=int(args.target_size),
        seed=int(args.seed),
        deepset_root=str(args.deepset_root),
        wainject_root=str(args.wainject_root),
        source_weights=source_weights,
        chunk_weights=chunk_weights,
        ordinal_bins=(float(ordinal_bins_vals[0]), float(ordinal_bins_vals[1]), float(ordinal_bins_vals[2])),
        profile=str(args.profile),
    )
    rows, selection_manifest = build_gold_slice_prefill(cfg=cfg, projector=projector)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(
        args.output_dir
        or (ROOT / "artifacts" / "gold_slice" / f"prefill_{run_id}_{snapshot.resolved_sha256[:12]}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    prefill_path = out_dir / "prefill_candidates.jsonl"
    write_jsonl(prefill_path.as_posix(), rows)
    write_json(
        (out_dir / "selection_manifest.json").as_posix(),
        {
            **selection_manifest,
            "resolved_config_sha256": snapshot.resolved_sha256,
            "profile": str(args.profile),
            "deepset_root": str(args.deepset_root),
            "wainject_root": str(args.wainject_root),
            "source_weights": source_weights,
            "chunk_weights": chunk_weights,
            "ordinal_bins": list(cfg.ordinal_bins),
            "prefill_path": prefill_path.as_posix(),
        },
    )
    if bool(args.write_annotator_seeds):
        write_jsonl((out_dir / "annotator_a.jsonl").as_posix(), rows)
        write_jsonl((out_dir / "annotator_b.jsonl").as_posix(), rows)

    report = {
        "status": "ok",
        "output_dir": out_dir.as_posix(),
        "prefill_candidates": prefill_path.as_posix(),
        "selection_manifest": (out_dir / "selection_manifest.json").as_posix(),
        "selected_size": int(len(rows)),
        "source_counts": selection_manifest.get("source_counts", {}),
        "chunk_counts": selection_manifest.get("chunk_counts", {}),
        "attack_count": selection_manifest.get("attack_count", 0),
        "benign_count": selection_manifest.get("benign_count", 0),
    }
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
