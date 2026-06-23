#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the real local Wave C multimodal production path on fixed documents.

This gate invokes production attachment extraction plus the bundled local_vision
RapidOCR+pi0 adapter on PDF pages and embedded DOCX/HTML images. It deliberately
disables region rescue so every original visual asset is processed exactly once;
Wave B separately gates spatial rescue. No cloud provider is called.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
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
from omega.vision.ocr_runtime import shutdown_ocr_workers

SUITE_VERSION = "vision_wave_c_local_v1"
KEY_IMPLEMENTATION_FILES = (
    "omega/rag/attachment_ingestion.py",
    "omega/rag/attachment_parser_runtime.py",
    "omega/rag/attachment_parser_broker.py",
    "omega/rag/attachment_sandbox_worker.py",
    "omega/projector/api_hybrid_projector.py",
    "omega/projector/api_hybrid/providers.py",
    "omega/projector/api_hybrid/semantic_contracts.py",
    "omega/projector/factory.py",
    "omega/vision/egress_policy.py",
    "omega/vision/ocr_worker.py",
    "omega/vision/ocr_runtime.py",
    "omega/config/validators/projector.py",
    "config/profiles/prod_vision.yml",
    "config/release_gate.yml",
    "omega/config/resources/profiles/prod_vision.yml",
    "omega/config/resources/release_gate.yml",
    "scripts/eval_vision_wave_c_local.py",
)
FORBIDDEN_RAW_MEDIA_KEYS = {
    "payload_b64",
    "bytes_b64",
    "raw_bytes",
    "file_bytes",
    "image_bytes",
    "image_bytes_b64",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    idx = max(0, min(len(rows) - 1, round((len(rows) - 1) * q)))
    return float(rows[idx])


def _raw_findings(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(int(str(key) in FORBIDDEN_RAW_MEDIA_KEYS) for key in value) + sum(
            _raw_findings(child) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_raw_findings(child) for child in value)
    return 0


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("invalid Wave C manifest")
    return rows


def _trace_from_projection(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    matches = getattr(getattr(result, "evidence", None), "matches", {})
    if not isinstance(matches, Mapping):
        return {}, {}
    api = matches.get("api_perception", {})
    if not isinstance(api, Mapping):
        return {}, {}
    trace = api.get("execution_trace", {})
    return dict(api), dict(trace) if isinstance(trace, Mapping) else {}


def evaluate(*, manifest_path: Path, profile: str) -> dict[str, Any]:
    rows = _load_manifest(manifest_path)
    fixture_root = manifest_path.parent.resolve()
    snapshot = load_resolved_config(profile=profile)
    cfg = copy.deepcopy(snapshot.resolved)
    gate_cfg = cfg["vision_wave_c"]["local_gate"]["gates"]

    # Fresh, request-isolated state: a committed cache must never influence a quality gate.
    scratch = Path(tempfile.mkdtemp(prefix="omega-wave-c-local-"))
    api_cfg = cfg["projector"]["api_perception"]
    api_cfg["cache_path"] = str(scratch / "cache.jsonl")
    api_cfg["error_log_path"] = str(scratch / "errors.jsonl")
    api_cfg["image_region_pass_enabled"] = False
    api_cfg.setdefault("image_region_pass", {})["enabled"] = False
    api_cfg["prewarm_on_init"] = True

    attachment_cfg = copy.deepcopy(cfg["retriever"]["sqlite_fts"]["attachments"])
    # The local semantic provider owns OCR in this gate. Parser OCR would duplicate work.
    attachment_cfg.setdefault("ocr", {})["enabled"] = "false"

    projector = build_projector(cfg)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    raw_findings = 0
    try:
        for row in rows:
            print(f"[wave-c-local] {row['id']}: extracting", flush=True)
            file_path = (fixture_root / str(row["file"])).resolve()
            if fixture_root not in file_path.parents or _sha(file_path) != str(
                row["sha256"]
            ):
                raise ValueError(f"fixture integrity failure: {row['id']}")

            extraction_started = time.perf_counter()
            extracted = extract_attachment(
                content_bytes=file_path.read_bytes(),
                filename=str(row["filename"]),
                mime=str(row["mime"]),
                cfg=attachment_cfg,
            )
            extraction_ms = (time.perf_counter() - extraction_started) * 1000.0
            assets = list(extracted.visual_assets or [])
            scope_id = f"wave-c-local:{row['id']}"
            variants: list[dict[str, Any]] = []
            manifest: list[dict[str, Any]] = []
            for asset in assets:
                blob_ref = projector.register_image_blob(
                    scope_id=scope_id,
                    data=asset.decode(),
                    mime=asset.mime,
                    expected_sha256=asset.sha256,
                )
                variants.append(
                    {
                        "mime": asset.mime,
                        "bytes_ref": blob_ref,
                        "sha256": asset.sha256,
                        "size_bytes": asset.size_bytes,
                        "width": asset.width,
                        "height": asset.height,
                        "role": asset.role,
                    }
                )
                manifest.append(
                    {
                        "asset_id": asset.asset_id,
                        "sha256": asset.sha256,
                        "source_kind": asset.source_kind,
                        "page_number": asset.page_number,
                        "embedded_index": asset.embedded_index,
                    }
                )

            semantic_image: dict[str, Any] | None
            if len(variants) == 1:
                semantic_image = variants[0]
            elif variants:
                semantic_image = {"variants": variants}
            else:
                semantic_image = None
            item_meta: dict[str, Any] = {
                "tenant_id": "benchmark",
                "data_region": "local",
                "request_id": scope_id,
                "visual_asset_manifest": manifest,
                "visual_status": extracted.visual_status,
            }
            if semantic_image is not None:
                item_meta["semantic_image"] = semantic_image
            raw_findings += _raw_findings(item_meta)

            item = ContentItem(
                doc_id=str(row["id"]),
                source_id=f"wave-c-local:{row['id']}",
                source_type=str(row["format"]),
                trust="untrusted",
                # Empty text is intentional: attachment OCR is disabled above,
                # so the local visual provider must OCR the registered assets once.
                text=str(extracted.text or ""),
                meta=item_meta,
            )
            inference_started = time.perf_counter()
            try:
                projected = projector.project(item)
            finally:
                projector.release_image_scope(scope_id)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0

            api, trace = _trace_from_projection(projected)
            target_wall = str(row.get("target_wall") or "")
            target_idx = (
                WALLS_V1.index(target_wall) if target_wall in WALLS_V1 else None
            )
            target_hit = bool(
                target_idx is not None and float(projected.v[target_idx]) > 0.0
            )
            any_pressure = bool(max(float(value) for value in projected.v) > 0.0)
            semantic_status = str(api.get("semantic_status", ""))
            egress_allowed = (
                str(trace.get("visual_egress_decision", "")) == "allow"
                and str(trace.get("provider", "")) == "local_vision"
                and str(trace.get("provider_processing_region", "")) == "local"
            )
            residency_traced = (
                str(trace.get("tenant_id", "")) == "benchmark"
                and str(trace.get("data_region", "")) == "local"
            )
            print(
                f"[wave-c-local] {row['id']}: status={semantic_status} pressure={list(map(float, projected.v))}",
                flush=True,
            )
            results.append(
                {
                    "id": str(row["id"]),
                    "label": str(row["label"]),
                    "format": str(row["format"]),
                    "target_wall": target_wall or None,
                    "asset_count": len(assets),
                    "expected_asset_count": int(row["expected_asset_count"]),
                    "visual_status": extracted.visual_status,
                    "semantic_status": semantic_status,
                    "target_wall_hit": target_hit,
                    "any_pressure": any_pressure,
                    "pressure": {
                        wall: float(projected.v[idx])
                        for idx, wall in enumerate(WALLS_V1)
                    },
                    "provider": trace.get("provider"),
                    "provider_route": trace.get("provider_route", []),
                    "egress_allowed_local": egress_allowed,
                    "residency_traced": residency_traced,
                    "extraction_ms": extraction_ms,
                    "inference_ms": inference_ms,
                }
            )
    finally:
        shutdown_ocr_workers()
        shutdown_attachment_parser_broker()
        shutil.rmtree(scratch, ignore_errors=True)

    attacks = [row for row in results if row["label"] == "malicious"]
    benign = [row for row in results if row["label"] == "benign"]
    expected_multi = [row for row in rows if bool(row.get("multi_image", False))]
    extraction_latencies = [float(row["extraction_ms"]) for row in results]
    inference_latencies = [float(row["inference_ms"]) for row in results]
    semantic_success_statuses = {"vision_semantic_active", "semantic_active"}
    summary = {
        "samples_total": len(results),
        "attack_samples": len(attacks),
        "benign_samples": len(benign),
        "extraction_success_rate": sum(
            row["visual_status"] == "success"
            and row["asset_count"] == row["expected_asset_count"]
            for row in results
        )
        / len(results),
        "semantic_success_rate": sum(
            row["semantic_status"] in semantic_success_statuses for row in results
        )
        / len(results),
        "attack_target_wall_recall": sum(
            bool(row["target_wall_hit"]) for row in attacks
        )
        / len(attacks),
        "benign_false_positive_rate": sum(bool(row["any_pressure"]) for row in benign)
        / len(benign),
        "local_egress_trace_rate": sum(
            bool(row["egress_allowed_local"]) for row in results
        )
        / len(results),
        "data_residency_trace_rate": sum(
            bool(row["residency_traced"]) for row in results
        )
        / len(results),
        "multi_image_packet_rate": sum(
            next(result for result in results if result["id"] == row["id"])[
                "asset_count"
            ]
            > 1
            for row in expected_multi
        )
        / len(expected_multi)
        if expected_multi
        else 1.0,
        "raw_media_boundary_findings": raw_findings,
        "latency_ms": {
            "extraction_p50": _percentile(extraction_latencies, 0.50),
            "extraction_p95": _percentile(extraction_latencies, 0.95),
            "inference_p50": _percentile(inference_latencies, 0.50),
            "inference_p95": _percentile(inference_latencies, 0.95),
            "inference_avg": statistics.fmean(inference_latencies),
        },
        "duration_sec": time.perf_counter() - started,
    }
    thresholds = {key: float(value) for key, value in gate_cfg.items()}
    gates = {
        "extraction_success_rate": summary["extraction_success_rate"]
        >= thresholds["extraction_success_rate_min"],
        "semantic_success_rate": summary["semantic_success_rate"]
        >= thresholds["semantic_success_rate_min"],
        "attack_target_wall_recall": summary["attack_target_wall_recall"]
        >= thresholds["attack_target_wall_recall_min"],
        "benign_false_positive_rate": summary["benign_false_positive_rate"]
        <= thresholds["benign_false_positive_rate_max"],
        "local_egress_trace_rate": summary["local_egress_trace_rate"]
        >= thresholds["local_egress_trace_rate_min"],
        "data_residency_trace_rate": summary["data_residency_trace_rate"]
        >= thresholds["data_residency_trace_rate_min"],
        "multi_image_packet_rate": summary["multi_image_packet_rate"]
        >= thresholds["multi_image_packet_rate_min"],
        "raw_media_boundary_findings": summary["raw_media_boundary_findings"]
        <= thresholds["raw_media_boundary_findings_max"],
    }
    try:
        import rapidocr

        rapidocr_version = str(getattr(rapidocr, "__version__", "unknown"))
    except Exception:
        rapidocr_version = "unknown"
    try:
        import onnxruntime

        onnxruntime_version = str(onnxruntime.__version__)
    except Exception:
        onnxruntime_version = "unknown"
    return {
        "suite_version": SUITE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "quality_claim": "local_fixed_fixture_multiformat_multi_image_quality_only",
        "profile": profile,
        "manifest_sha256": _sha(manifest_path),
        "implementation_hashes": {
            rel: _sha(ROOT / rel) for rel in KEY_IMPLEMENTATION_FILES
        },
        "provider": {
            "type": "local_vision",
            "backend": "ocr_pi0",
            "rapidocr": rapidocr_version,
            "onnxruntime": onnxruntime_version,
        },
        "thresholds": thresholds,
        "summary": summary,
        "gates": gates,
        "results": results,
        "limitations": [
            "Fixed-fixture local quality gate; not a representative production traffic claim.",
            "Spatial rescue is disabled here to avoid duplicate OCR; Wave B gates region rescue separately.",
            "OpenAI and Anthropic are not called by this gate.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="tests/data/vision_wave_c_frozen/manifest.jsonl"
    )
    parser.add_argument("--profile", default="prod_vision")
    parser.add_argument(
        "--output", default="artifacts/vision_wave_c/local/vision_wave_c_local_v1.json"
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate(manifest_path=manifest, profile=str(args.profile))
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "summary": report["summary"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
