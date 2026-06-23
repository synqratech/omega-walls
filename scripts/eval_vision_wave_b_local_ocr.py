#!/usr/bin/env python3
"""Run the real local RapidOCR production path on the fixed Wave B corpus.

Unlike the network-free fusion gate, this invokes the installed OCR engine through
Omega's persistent worker and the sandboxed attachment path. It does not call a
cloud vision provider and does not claim representative real-world OCR quality.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from omega.rag.attachment_ingestion import extract_attachment
from omega.vision.ocr_runtime import OCRWorkerSettings, prewarm_ocr_worker, shutdown_ocr_workers

SUITE_VERSION = "vision_wave_b_local_rapidocr_v1"
KEY_IMPLEMENTATION_FILES = (
    "omega/vision/ocr_worker.py",
    "omega/vision/ocr_runtime.py",
    "omega/vision/contracts.py",
    "omega/rag/attachment_ingestion.py",
    "omega/projector/pi0_intent_v2.py",
    "omega/projector/pi0_v2/secret_signals.py",
    "omega/api/server.py",
    "config/pi0_defaults.yml",
    "config/retriever.yml",
    "config/profiles/prod_vision_local_ocr.yml",
    "config/release_gate.yml",
    "omega/config/resources/pi0_defaults.yml",
    "omega/config/resources/retriever.yml",
    "omega/config/resources/profiles/prod_vision_local_ocr.yml",
    "omega/config/resources/release_gate.yml",
    "omega/config/validators/release_gate.py",
    "pyproject.toml",
    "scripts/eval_vision_wave_b_local_ocr.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    rows = sorted(values)
    idx = max(0, min(len(rows) - 1, round((len(rows) - 1) * q)))
    return float(rows[idx])


def _expected_text(row: dict[str, Any]) -> str:
    spans = list(row.get("ocr_spans", []) or [])
    trusted = [str(span.get("text", "")) for span in spans if float(span.get("confidence", 0.0)) >= 0.55]
    return " ".join(trusted).strip()


def evaluate(*, manifest: Path, profile: str) -> dict[str, Any]:
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    snapshot = load_resolved_config(profile=profile)
    cfg = snapshot.resolved
    attachment_cfg = cfg["retriever"]["sqlite_fts"]["attachments"]
    ocr_cfg = attachment_cfg["ocr"]
    gate_cfg = cfg["vision_wave_b"]["local_rapidocr_gate"]["gates"]
    pi0_cfg = copy.deepcopy(cfg)
    pi0_cfg["pi0"]["semantic"]["enabled"] = "false"
    pi0 = Pi0IntentAwareV2(pi0_cfg)
    settings = OCRWorkerSettings(
        provider=str(ocr_cfg["provider"]),
        startup_timeout_sec=float(ocr_cfg["worker_startup_timeout_sec"]),
        request_timeout_sec=float(ocr_cfg["worker_request_timeout_sec"]),
        max_memory_mb=int(ocr_cfg["worker_max_memory_mb"]),
        max_requests_per_worker=int(ocr_cfg["worker_max_requests"]),
        pool_size=int(ocr_cfg.get("worker_pool_size", 1)),
        max_pending_requests=int(ocr_cfg.get("worker_max_pending_requests", 2)),
        queue_timeout_sec=float(ocr_cfg.get("worker_queue_timeout_sec", 1.0)),
        intra_op_num_threads=int(ocr_cfg.get("worker_intra_op_threads", 2)),
        inter_op_num_threads=int(ocr_cfg.get("worker_inter_op_threads", 1)),
    )
    started = time.perf_counter()
    prewarm_started = time.perf_counter()
    prewarm_ocr_worker(settings)
    prewarm_ms = (time.perf_counter() - prewarm_started) * 1000.0
    results: list[dict[str, Any]] = []
    try:
        for row in rows:
            path = manifest.parent / str(row["file"])
            request_started = time.perf_counter()
            extracted = extract_attachment(
                content_bytes=path.read_bytes(),
                filename=path.name,
                mime=str(row["mime"]),
                cfg=attachment_cfg,
            )
            latency_ms = (time.perf_counter() - request_started) * 1000.0
            projection = pi0.project(
                ContentItem(
                    doc_id=str(row["id"]),
                    source_id=f"wave-b-local:{row['id']}",
                    source_type="image",
                    trust="untrusted",
                    text=extracted.text,
                )
            )
            target_wall = str(row.get("target_wall") or "")
            target_idx = WALLS_V1.index(target_wall) if target_wall in WALLS_V1 else None
            target_hit = bool(target_idx is not None and float(projection.v[target_idx]) > 0.0)
            any_pressure = bool(float(projection.v.max()) > 0.0)
            expected_text = _expected_text(row)
            similarity = float(fuzz.ratio(expected_text.casefold(), extracted.text.casefold()) / 100.0) if expected_text else 1.0
            results.append(
                {
                    "id": str(row["id"]),
                    "label": str(row["label"]),
                    "target_wall": target_wall or None,
                    "ocr_status": extracted.ocr_status,
                    "ocr_text": extracted.text,
                    "expected_text": expected_text,
                    "text_similarity": similarity,
                    "target_wall_hit": target_hit,
                    "any_pressure": any_pressure,
                    "pressure": {wall: float(projection.v[idx]) for idx, wall in enumerate(WALLS_V1)},
                    "ocr_quality": extracted.ocr_quality.to_dict(),
                    "latency_ms": latency_ms,
                }
            )
    finally:
        shutdown_ocr_workers()

    attacks = [row for row in results if row["label"] == "malicious"]
    benign = [row for row in results if row["label"] == "benign"]
    latencies = [float(row["latency_ms"]) for row in results]
    summary = {
        "samples_total": len(results),
        "attack_samples": len(attacks),
        "benign_samples": len(benign),
        "ocr_success_rate": sum(row["ocr_status"] == "success" for row in results) / len(results),
        "text_similarity_ge_0_80_rate": sum(float(row["text_similarity"]) >= 0.80 for row in results) / len(results),
        "attack_target_wall_recall": sum(bool(row["target_wall_hit"]) for row in attacks) / len(attacks),
        "benign_false_positive_rate": sum(bool(row["any_pressure"]) for row in benign) / len(benign),
        "quality_usable_rate": sum(row["ocr_quality"]["status"] == "usable" for row in results) / len(results),
        "prewarm_ms": prewarm_ms,
        "latency_ms": {
            "avg": statistics.fmean(latencies),
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "duration_sec": time.perf_counter() - started,
    }
    thresholds = {key: float(value) for key, value in gate_cfg.items()}
    gates = {
        "ocr_success_rate": summary["ocr_success_rate"] >= thresholds["ocr_success_rate_min"],
        "text_similarity_rate": summary["text_similarity_ge_0_80_rate"] >= thresholds["text_similarity_rate_min"],
        "attack_target_wall_recall": summary["attack_target_wall_recall"] >= thresholds["attack_target_wall_recall_min"],
        "benign_false_positive_rate": summary["benign_false_positive_rate"] <= thresholds["benign_false_positive_rate_max"],
        "quality_usable_rate": summary["quality_usable_rate"] >= thresholds["quality_usable_rate_min"],
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
        "quality_claim": "local_fixed_fixture_rapidocr_quality_only",
        "profile": profile,
        "manifest_sha256": _sha(manifest),
        "implementation_hashes": {rel: _sha(ROOT / rel) for rel in KEY_IMPLEMENTATION_FILES},
        "provider": {"rapidocr": rapidocr_version, "onnxruntime": onnxruntime_version},
        "summary": summary,
        "thresholds": thresholds,
        "gates": gates,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="tests/data/vision_wave_b_frozen/manifest.jsonl")
    parser.add_argument("--profile", default="prod_vision_local_ocr")
    parser.add_argument("--output", default="artifacts/vision_wave_b/local_rapidocr/vision_wave_b_local_rapidocr_v1.json")
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    if not manifest.is_absolute(): manifest = ROOT / manifest
    if not output.is_absolute(): output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate(manifest=manifest, profile=str(args.profile))
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"], "output": str(output)}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
