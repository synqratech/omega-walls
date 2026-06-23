#!/usr/bin/env python3
# ruff: noqa: E402
"""Credentialed live benchmark for a pinned cloud multimodal provider/model.

This script intentionally requires an explicit external-egress acknowledgement.
It emits a report but is not run in normal CI because provider credentials and
pinned model availability are deployment-specific.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.projector.factory import build_projector
from omega.rag.attachment_ingestion import extract_attachment
from omega.rag.attachment_parser_runtime import shutdown_attachment_parser_broker


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pct(values: list[float], q: float) -> float:
    rows = sorted(values)
    if not rows:
        return 0.0
    return float(rows[max(0, min(len(rows) - 1, round((len(rows) - 1) * q)))])


def _api_trace(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = getattr(getattr(result, "evidence", None), "matches", {})
    api = matches.get("api_perception", {}) if isinstance(matches, Mapping) else {}
    trace = api.get("execution_trace", {}) if isinstance(api, Mapping) else {}
    return dict(api) if isinstance(api, Mapping) else {}, dict(trace) if isinstance(
        trace, Mapping
    ) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=("openai", "anthropic"), required=True)
    parser.add_argument("--model", required=True, help="Pinned provider model id.")
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--manifest", default="tests/data/vision_wave_c_frozen/manifest.jsonl"
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--acknowledge-external-visual-egress", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_external_visual_egress:
        raise SystemExit(
            "Refusing cloud visual egress without --acknowledge-external-visual-egress"
        )
    if not os.environ.get(args.api_key_env, "").strip():
        raise SystemExit(f"Missing provider credential env: {args.api_key_env}")

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    fixture_root = manifest_path.parent.resolve()
    cfg = copy.deepcopy(load_resolved_config(profile="pilot").resolved)
    scratch = Path(tempfile.mkdtemp(prefix="omega-wave-c-cloud-"))
    api_cfg = cfg["projector"]["api_perception"]
    api_cfg.update(
        {
            "provider": args.provider,
            "model": args.model,
            "api_key_env": args.api_key_env,
            "cache_path": str(scratch / "cache.jsonl"),
            "error_log_path": str(scratch / "errors.jsonl"),
            "image_region_pass_enabled": False,
        }
    )
    api_cfg["image_region_pass"]["enabled"] = False
    if args.base_url:
        api_cfg["base_url"] = args.base_url
    elif args.provider == "anthropic":
        api_cfg["base_url"] = "https://api.anthropic.com/v1"
    else:
        api_cfg["base_url"] = "https://api.openai.com/v1"
    api_cfg["provider_options"]["visual_egress"] = {
        "enabled": True,
        "default_action": "deny",
        "providers": {args.provider: {"external": True, "region": "global"}},
        "tenants": {
            "benchmark": {
                "allow_external": True,
                "allowed_providers": [args.provider],
                "allowed_regions": ["global"],
                "require_region_match": False,
            }
        },
    }
    api_cfg["provider_options"]["capabilities"] = {
        "text": True,
        "image": True,
        "supported_image_mime_types": [
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif",
        ],
        "max_image_bytes": 20 * 1024 * 1024,
        "max_images": 8,
    }
    attachment_cfg = copy.deepcopy(cfg["retriever"]["sqlite_fts"]["attachments"])
    attachment_cfg["ocr"]["enabled"] = "false"
    projector = build_projector(cfg)
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for row in rows:
            path = (fixture_root / str(row["file"])).resolve()
            if _sha(path) != str(row["sha256"]):
                raise ValueError(f"fixture integrity failure: {row['id']}")
            extracted = extract_attachment(
                content_bytes=path.read_bytes(),
                filename=str(row["filename"]),
                mime=str(row["mime"]),
                cfg=attachment_cfg,
            )
            scope = f"wave-c-cloud:{row['id']}"
            variants = []
            for asset in extracted.visual_assets:
                ref = projector.register_image_blob(
                    scope_id=scope,
                    data=asset.decode(),
                    mime=asset.mime,
                    expected_sha256=asset.sha256,
                )
                variants.append(
                    {
                        "mime": asset.mime,
                        "bytes_ref": ref,
                        "sha256": asset.sha256,
                        "size_bytes": asset.size_bytes,
                        "width": asset.width,
                        "height": asset.height,
                        "role": asset.role,
                    }
                )
            semantic_image = (
                variants[0] if len(variants) == 1 else {"variants": variants}
            )
            item = ContentItem(
                doc_id=str(row["id"]),
                source_id=f"cloud-benchmark:{row['id']}",
                source_type=str(row["format"]),
                trust="untrusted",
                text=str(extracted.text or "[attachment_visual_only]"),
                meta={
                    "tenant_id": "benchmark",
                    "data_region": "global",
                    "request_id": scope,
                    "semantic_image": semantic_image,
                },
            )
            t0 = time.perf_counter()
            try:
                projected = projector.project(item)
            finally:
                projector.release_image_scope(scope)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            api, trace = _api_trace(projected)
            target = str(row.get("target_wall") or "")
            target_idx = WALLS_V1.index(target) if target in WALLS_V1 else None
            results.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "format": row["format"],
                    "target_wall": target or None,
                    "target_hit": bool(
                        target_idx is not None and float(projected.v[target_idx]) > 0
                    ),
                    "any_pressure": bool(max(map(float, projected.v)) > 0),
                    "pressure": {
                        wall: float(projected.v[i]) for i, wall in enumerate(WALLS_V1)
                    },
                    "semantic_status": api.get("semantic_status"),
                    "provider": trace.get("provider"),
                    "provider_id": trace.get("provider_id"),
                    "provider_route": trace.get("provider_route", []),
                    "token_usage": trace.get("token_usage", {}),
                    "latency_ms": latency_ms,
                }
            )
    finally:
        shutdown_attachment_parser_broker()
        shutil.rmtree(scratch, ignore_errors=True)

    attacks = [row for row in results if row["label"] == "malicious"]
    benign = [row for row in results if row["label"] == "benign"]
    latencies = [float(row["latency_ms"]) for row in results]
    summary = {
        "samples_total": len(results),
        "attack_target_wall_recall": sum(bool(row["target_hit"]) for row in attacks)
        / len(attacks),
        "benign_false_positive_rate": sum(bool(row["any_pressure"]) for row in benign)
        / len(benign),
        "semantic_success_rate": sum(
            row["semantic_status"] in {"semantic_active", "vision_semantic_active"}
            for row in results
        )
        / len(results),
        "latency_ms_p50": _pct(latencies, 0.50),
        "latency_ms_p95": _pct(latencies, 0.95),
        "latency_ms_avg": statistics.fmean(latencies),
        "duration_sec": time.perf_counter() - started,
    }
    report = {
        "suite_version": "vision_wave_c_cloud_live_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": args.provider,
        "model": args.model,
        "base_url": api_cfg["base_url"],
        "manifest_sha256": _sha(manifest_path),
        "quality_claim": "credentialed_fixed_fixture_cloud_provider_measurement",
        "summary": summary,
        "results": results,
    }
    output = Path(
        args.output
        or f"artifacts/vision_wave_c/cloud_live/{args.provider}-{args.model.replace('/', '_')}.json"
    )
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(output), "summary": summary}, ensure_ascii=False, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
