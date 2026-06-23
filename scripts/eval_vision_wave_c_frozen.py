#!/usr/bin/env python3
# ruff: noqa: E402
"""Network-free production contract benchmark for Vision Wave C."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem
from omega.projector.api_hybrid.providers import capabilities_for_provider
from omega.projector.api_hybrid_projector import APIPerceptionProjector
from omega.rag.attachment_ingestion import extract_attachment
from omega.vision.egress_policy import VisualEgressPolicy

SUITE_VERSION = "vision_wave_c_frozen_v1"
KEY_IMPLEMENTATION_FILES = (
    "omega/rag/attachment_ingestion.py",
    "omega/rag/attachment_parser_runtime.py",
    "omega/rag/attachment_parser_broker.py",
    "omega/rag/attachment_sandbox_worker.py",
    "omega/api/scan_request_orchestration.py",
    "omega/api/request_parsing.py",
    "omega/api/openapi_models.py",
    "omega/api/server.py",
    "omega/sdk.py",
    "omega/rag/harness.py",
    "omega/vision/egress_policy.py",
    "omega/projector/api_hybrid/semantic_contracts.py",
    "omega/projector/api_hybrid/providers.py",
    "omega/projector/api_hybrid_projector.py",
    "omega/config/validators/projector.py",
    "config/profiles/prod_vision.yml",
    "omega/config/resources/profiles/prod_vision.yml",
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


def _load(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows or len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("invalid Wave C manifest")
    return rows


def _raw_findings(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(1 for key in value if str(key) in FORBIDDEN_RAW_MEDIA_KEYS) + sum(
            _raw_findings(v) for v in value.values()
        )
    if isinstance(value, (list, tuple)):
        return sum(_raw_findings(v) for v in value)
    return 0


def _projector_cfg(tmp_root: Path, provider: str) -> dict[str, Any]:
    external = provider != "local_vision"
    region = "global" if external else "local"
    return {
        "projector": {
            "mode": "hybrid_api",
            "api_perception": {
                "enabled": "true",
                "strict": True,
                "semantic_mode": "local_semantic"
                if provider == "local_vision"
                else "hybrid_cloud",
                "provider": provider,
                "model": "wave-c-contract",
                "base_url": "local://vision"
                if provider == "local_vision"
                else (
                    "https://api.anthropic.com/v1"
                    if provider == "anthropic"
                    else "https://api.openai.com/v1"
                ),
                "api_key_env": "VISION_WAVE_C_TEST_KEY",
                "cache_path": str(tmp_root / f"{provider}-cache.jsonl"),
                "error_log_path": str(tmp_root / f"{provider}-errors.jsonl"),
                "provider_options": {
                    "capabilities": {
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
                    },
                    "local_vision": {"backend": "ocr_pi0"},
                    "visual_egress": {
                        "enabled": True,
                        "default_action": "deny",
                        "providers": {
                            provider: {"external": external, "region": region}
                        },
                        "tenants": {
                            "benchmark": {
                                "allow_external": external,
                                "allowed_providers": [provider],
                                "allowed_regions": [region],
                                "require_region_match": False,
                            }
                        },
                    },
                },
            },
        },
    }


def evaluate(*, manifest_path: Path, output_path: Path) -> dict[str, Any]:
    rows = _load(manifest_path)
    root = manifest_path.parent.resolve()
    tested_profile = "prod_vision"
    prod = load_resolved_config(profile=tested_profile).resolved
    attachment_cfg = dict(prod["retriever"]["sqlite_fts"]["attachments"])
    # Deterministic contract gate uses in-process parsers; sandbox behavior is
    # covered by the security suite and the local production benchmark.
    attachment_cfg["sandbox"] = {"enabled": False}
    attachment_cfg["ocr"] = {"enabled": "false"}

    os.environ["VISION_WAVE_C_TEST_KEY"] = "wave-c-test-key-123456789"
    tmp_root = output_path.parent / ".wave-c-contract-tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    projectors = {
        provider: APIPerceptionProjector(_projector_cfg(tmp_root, provider))
        for provider in ("openai", "anthropic", "local_vision")
    }

    exact_assets = 0
    provenance_ok = 0
    integrity_ok = 0
    multi_packets = 0
    expected_multi = 0
    remote_reference_ignored = 0
    raw_findings = 0
    extraction_ms: list[float] = []
    sample_rows: list[dict[str, Any]] = []

    for row in rows:
        file_path = (root / str(row["file"])).resolve()
        if root not in file_path.parents or _sha(file_path) != str(row["sha256"]):
            raise ValueError(f"fixture integrity failure: {row['id']}")
        started = time.perf_counter()
        extracted = extract_attachment(
            content_bytes=file_path.read_bytes(),
            filename=str(row["filename"]),
            mime=str(row["mime"]),
            cfg=attachment_cfg,
        )
        extraction_ms.append((time.perf_counter() - started) * 1000.0)
        assets = list(extracted.visual_assets or [])
        count_ok = len(assets) == int(row["expected_asset_count"])
        exact_assets += int(count_ok)
        kinds_ok = [str(asset.source_kind) for asset in assets] == list(
            row["expected_source_kinds"]
        )
        provenance_ok += int(kinds_ok)
        digest_ok = all(
            hashlib.sha256(asset.decode()).hexdigest() == asset.sha256
            for asset in assets
        )
        integrity_ok += int(digest_ok)
        if bool(row.get("multi_image", False)):
            expected_multi += 1
            multi_packets += int(len(assets) > 1)
        if bool(row.get("contains_remote_image_reference", False)):
            remote_reference_ignored += int(
                len(assets) == int(row["expected_asset_count"])
            )

        provider_support: dict[str, bool] = {}
        for provider, projector in projectors.items():
            variants = []
            for asset in assets:
                ref = projector.register_image_blob(
                    scope_id=str(row["id"]),
                    data=asset.decode(),
                    mime=asset.mime,
                    expected_sha256=asset.sha256,
                )
                variants.append(
                    {
                        "mime": asset.mime,
                        "sha256": asset.sha256,
                        "bytes_ref": ref,
                        "size_bytes": asset.size_bytes,
                        "width": asset.width,
                        "height": asset.height,
                        "role": asset.role,
                    }
                )
            item = ContentItem(
                doc_id=str(row["id"]),
                source_id=f"wave-c:{row['id']}",
                source_type=str(row["format"]),
                trust="untrusted",
                text=str(extracted.text or "[attachment_visual_only]"),
                meta={
                    "tenant_id": "benchmark",
                    "data_region": "local" if provider == "local_vision" else "global",
                    "request_id": str(row["id"]),
                    "semantic_image": variants[0]
                    if len(variants) == 1
                    else {"variants": variants},
                    "visual_asset_manifest": [
                        {
                            "asset_id": asset.asset_id,
                            "sha256": asset.sha256,
                            "source_kind": asset.source_kind,
                            "page_number": asset.page_number,
                            "embedded_index": asset.embedded_index,
                        }
                        for asset in assets
                    ],
                },
            )
            semantic, source_meta = projector._build_semantic_input(
                item=item, text=item.text
            )
            provider_support[provider] = capabilities_for_provider(
                provider
            ).supports_input(semantic)
            raw_findings += _raw_findings(item.meta) + _raw_findings(source_meta)
            projector.release_image_scope(str(row["id"]))

        sample_rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "format": row["format"],
                "asset_count": len(assets),
                "count_ok": count_ok,
                "provenance_ok": kinds_ok,
                "integrity_ok": digest_ok,
                "provider_support": provider_support,
                "visual_status": extracted.visual_status,
            }
        )

    egress = VisualEgressPolicy(
        {
            "enabled": True,
            "default_action": "deny",
            "providers": {
                "openai-eu": {"external": True, "region": "eu"},
                "openai-us": {"external": True, "region": "us"},
                "local_vision": {"external": False, "region": "local"},
            },
            "tenants": {
                "eu": {
                    "allow_external": True,
                    "allowed_providers": ["openai-eu"],
                    "allowed_regions": ["eu"],
                    "require_region_match": True,
                    "require_data_region": True,
                },
                "private": {
                    "allow_external": False,
                    "allowed_providers": ["local_vision"],
                    "allowed_regions": ["local"],
                },
            },
        }
    )
    egress_cases = {
        "eu_allowed": egress.decide(
            tenant_id="eu",
            data_region="eu",
            provider_id="openai-eu",
            provider_type="openai",
        ).allowed,
        "eu_us_denied": not egress.decide(
            tenant_id="eu",
            data_region="eu",
            provider_id="openai-us",
            provider_type="openai",
        ).allowed,
        "eu_unspecified_denied": not egress.decide(
            tenant_id="eu",
            data_region="unspecified",
            provider_id="openai-eu",
            provider_type="openai",
        ).allowed,
        "private_external_denied": not egress.decide(
            tenant_id="private",
            data_region="local",
            provider_id="openai-eu",
            provider_type="openai",
        ).allowed,
        "private_local_allowed": egress.decide(
            tenant_id="private",
            data_region="local",
            provider_id="local_vision",
            provider_type="local_vision",
        ).allowed,
    }
    sample_count = len(rows)
    summary = {
        "samples": sample_count,
        "formats": sorted({str(row["format"]) for row in rows}),
        "exact_asset_count_rate": exact_assets / sample_count,
        "provenance_accuracy": provenance_ok / sample_count,
        "asset_sha_integrity_rate": integrity_ok / sample_count,
        "multi_image_packet_rate": multi_packets / expected_multi
        if expected_multi
        else 1.0,
        "remote_reference_ignored_rate": remote_reference_ignored
        / sum(bool(row.get("contains_remote_image_reference")) for row in rows),
        "provider_capability_parity_rate": sum(
            all(bool(value) for value in sample["provider_support"].values())
            for sample in sample_rows
        )
        / sample_count,
        "raw_media_boundary_findings": raw_findings,
        "egress_policy_cases_passed": sum(bool(v) for v in egress_cases.values()),
        "egress_policy_cases_total": len(egress_cases),
        "extraction_latency_ms_median": statistics.median(extraction_ms),
        "extraction_latency_ms_p95": sorted(extraction_ms)[
            max(0, int(len(extraction_ms) * 0.95) - 1)
        ],
        "prod_projector_mode": str(prod["projector"]["mode"]),
        "prod_provider": str(prod["projector"]["api_perception"]["provider"]),
    }
    gates = {
        "asset_counts": summary["exact_asset_count_rate"] == 1.0,
        "provenance": summary["provenance_accuracy"] == 1.0,
        "sha_integrity": summary["asset_sha_integrity_rate"] == 1.0,
        "multi_image": summary["multi_image_packet_rate"] == 1.0,
        "remote_fetch_forbidden": summary["remote_reference_ignored_rate"] == 1.0,
        "provider_parity": summary["provider_capability_parity_rate"] == 1.0,
        "media_boundary": raw_findings == 0,
        "egress": all(egress_cases.values()),
        "prod_runtime_enabled": summary["prod_projector_mode"] == "hybrid_api"
        and summary["prod_provider"] == "local_vision",
    }
    report = {
        "suite_version": SUITE_VERSION,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "generated_at": __import__("datetime")
        .datetime.now(__import__("datetime").timezone.utc)
        .isoformat(),
        "manifest_sha256": _sha(manifest_path),
        "implementation_hashes": {
            rel: _sha(ROOT / rel) for rel in KEY_IMPLEMENTATION_FILES
        },
        "tested_profile": tested_profile,
        "summary": summary,
        "gates": gates,
        "egress_cases": egress_cases,
        "samples": sample_rows,
        "quality_claim": "Network-free Wave C architecture, extraction, multi-image, provider-capability, egress and residency contract gate; not a cloud-model quality claim.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", default="tests/data/vision_wave_c_frozen/manifest.jsonl"
    )
    parser.add_argument(
        "--output",
        default="artifacts/vision_wave_c/frozen/vision_wave_c_frozen_v1.json",
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not output.is_absolute():
        output = ROOT / output
    report = evaluate(manifest_path=manifest, output_path=output)
    print(
        json.dumps(
            {"status": report["status"], "summary": report["summary"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
