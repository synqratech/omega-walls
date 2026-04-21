from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze pilot baseline config + hash manifest")
    parser.add_argument("--profile", default="pilot_canonical")
    parser.add_argument("--output-root", default="artifacts/pilot_canonical")
    parser.add_argument("--label", default="baseline")
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=args.profile)
    run_id = f"{args.label}_{_utc_stamp()}_{snapshot.resolved_sha256[:12]}"
    out_dir = ROOT / args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    resolved_json = json.dumps(snapshot.resolved, ensure_ascii=True, indent=2, sort_keys=True, default=str)
    resolved_path = out_dir / "resolved_config.json"
    resolved_path.write_text(resolved_json, encoding="utf-8")

    manifest: Dict[str, Any] = {
        "event": "pilot_baseline_freeze_v1",
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "profile": args.profile,
        "resolved_config_sha256": snapshot.resolved_sha256,
        "resolved_config_file_sha256": _sha256_bytes(resolved_json.encode("utf-8")),
        "file_hashes": dict(snapshot.file_hashes),
        "guardrails": {
            "math_core_changed": False,
            "scope": "orchestration_policy_action_semantics_only",
            "canonical_path": "RAG service -> RetrieverProdAdapter -> OmegaRAGHarness -> ToolGateway",
        },
    }
    manifest_path = out_dir / "baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    latest = ROOT / args.output_root / "LATEST.json"
    latest.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "profile": args.profile,
                "baseline_manifest": str(manifest_path.relative_to(ROOT).as_posix()),
                "resolved_config": str(resolved_path.relative_to(ROOT).as_posix()),
            },
            ensure_ascii=True,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "run_id": run_id,
                "baseline_manifest": str(manifest_path.relative_to(ROOT).as_posix()),
                "resolved_config": str(resolved_path.relative_to(ROOT).as_posix()),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
