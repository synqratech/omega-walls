#!/usr/bin/env python3
"""Evaluate the frozen Phase 1 multimodal contract corpus.

This gate is deterministic and network-free. It exercises the real BlobRef boundary,
provider-agnostic SemanticInput, OpenAI adapter-facing path, strict result mapping,
ProjectionResult mapping, and request-scoped execution trace using recorded provider
results. It is a regression/contract gate, not a claim about current live model quality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from types import MethodType
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.interfaces.contracts_v1 import ContentItem, WALLS_V1  # noqa: E402
from omega.projector.api_hybrid.semantic_contracts import SemanticInput, SemanticResult  # noqa: E402
from omega.projector.api_hybrid_projector import APIPerceptionProjector  # noqa: E402

SUITE_VERSION = "vision_phase1_frozen_v1"
KEY_IMPLEMENTATION_FILES = (
    "omega/projector/api_hybrid/blob_store.py",
    "omega/projector/api_hybrid/semantic_contracts.py",
    "omega/projector/api_hybrid/providers.py",
    "omega/projector/api_hybrid_projector.py",
    "omega/api/chunk_pipeline.py",
    "omega/api/openapi_models.py",
    "omega/api/routes/scan.py",
    "omega/api/scan_request_orchestration.py",
)
FORBIDDEN_RAW_KEYS = {
    "image_bytes_b64",
    "image_variants",
    "bytes_b64",
    "raw_bytes",
    "file_bytes",
    "image_bytes",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _load_manifest(path: Path) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if not isinstance(row, dict):
            raise ValueError(f"manifest line {line_no} must be an object")
        sample_id = str(row.get("id", "")).strip()
        if not sample_id or sample_id in seen:
            raise ValueError(f"invalid or duplicate sample id at line {line_no}: {sample_id}")
        seen.add(sample_id)
        rows.append(dict(row))
    if not rows:
        raise ValueError("vision manifest is empty")
    return rows


def _config(tmp_dir: Path) -> Dict[str, Any]:
    return {
        "projector": {
            "api_perception": {
                "enabled": "true",
                "strict": True,
                "provider": "openai",
                "model": "frozen-recorded-openai-contract",
                "base_url": "https://api.openai.invalid/v1",
                "api_key_env": "OMEGA_FROZEN_VISION_TEST_KEY",
                "cache_path": str(tmp_dir / "cache.jsonl"),
                "error_log_path": str(tmp_dir / "errors.jsonl"),
                "semantic_mode": "hybrid_cloud",
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
                    "blob_ttl_sec": 30,
                },
            }
        }
    }


def _walk_forbidden(value: Any, *, path: str = "root") -> list[str]:
    findings: list[str] = []
    if isinstance(value, (bytes, bytearray, memoryview)):
        findings.append(f"{path}:raw_bytes_value")
    elif isinstance(value, Mapping):
        for key, nested in value.items():
            key_s = str(key)
            if key_s in FORBIDDEN_RAW_KEYS:
                findings.append(f"{path}.{key_s}:forbidden_key")
            findings.extend(_walk_forbidden(nested, path=f"{path}.{key_s}"))
    elif isinstance(value, (list, tuple)):
        for idx, nested in enumerate(value):
            findings.extend(_walk_forbidden(nested, path=f"{path}[{idx}]"))
    return findings


def _implementation_hashes() -> Dict[str, str]:
    return {name: _sha256_file(ROOT / name) for name in KEY_IMPLEMENTATION_FILES}


def evaluate(*, manifest_path: Path, output_path: Path) -> Dict[str, Any]:
    manifest_path = manifest_path.resolve()
    rows = _load_manifest(manifest_path)
    manifest_root = manifest_path.parent
    payload_by_sha: Dict[str, Dict[str, Any]] = {}
    dataset_rows: list[Dict[str, Any]] = []
    for row in rows:
        image_path = (manifest_root / str(row["file"])).resolve()
        if manifest_root.resolve() not in image_path.parents:
            raise ValueError(f"sample path escapes manifest root: {row['id']}")
        raw = image_path.read_bytes()
        actual_sha = _sha256_bytes(raw)
        expected_sha = str(row.get("sha256", "")).lower()
        if actual_sha != expected_sha:
            raise ValueError(f"image sha mismatch for {row['id']}")
        payload = dict(row.get("recorded_openai_result", {}) or {})
        SemanticResult.from_payload(
            payload=payload,
            semantic_status="vision_semantic_active",
            provider_meta={"provider": "openai", "mode": "recorded_frozen_contract"},
            vision_meta={"attempted": True, "provider_supported": True},
        )
        payload_by_sha[actual_sha] = payload
        dataset_rows.append(
            {
                "id": str(row["id"]),
                "image_sha256": actual_sha,
                "family": str(row.get("family", "")),
                "expected_nonzero": sorted(str(x) for x in row.get("expected_nonzero", [])),
                "expected_all_zero": bool(row.get("expected_all_zero", False)),
                "recorded_result_sha256": _canonical_sha(payload),
            }
        )

    old_key = os.environ.get("OMEGA_FROZEN_VISION_TEST_KEY")
    os.environ["OMEGA_FROZEN_VISION_TEST_KEY"] = "frozen-test-key-not-a-live-secret"
    sample_results: list[Dict[str, Any]] = []
    provider_inputs: Dict[str, SemanticInput] = {}
    try:
        with tempfile.TemporaryDirectory(prefix="omega-vision-frozen-") as tmp:
            projector = APIPerceptionProjector(config=_config(Path(tmp)))

            def _recorded_call(self, *, semantic_input=None, text=None):
                _ = text
                if not isinstance(semantic_input, SemanticInput):
                    raise TypeError("frozen gate requires SemanticInput")
                if len(semantic_input.image_parts) != 1:
                    raise ValueError("frozen gate expects exactly one image")
                image = semantic_input.image_parts[0]
                provider_inputs[str(image.sha256)] = semantic_input
                payload = payload_by_sha.get(str(image.sha256))
                if payload is None:
                    raise KeyError(f"missing frozen provider result for {image.sha256}")
                return dict(payload), f"frozen:{str(image.sha256)[:12]}", 0

            projector._call_api_scores = MethodType(_recorded_call, projector)  # type: ignore[method-assign]

            for row in rows:
                image_path = manifest_root / str(row["file"])
                raw = image_path.read_bytes()
                digest = _sha256_bytes(raw)
                scope_id = f"frozen-{row['id']}"
                ref = projector.register_image_blob(
                    scope_id=scope_id,
                    data=raw,
                    mime=str(row["mime"]),
                    expected_sha256=digest,
                )
                item = ContentItem(
                    doc_id=str(row["id"]),
                    source_id=f"vision-frozen:{row['id']}",
                    source_type="image",
                    trust="untrusted",
                    text="",
                    meta={
                        "request_id": scope_id,
                        "semantic_image": {
                            "mime": str(row["mime"]),
                            "sha256": digest,
                            "bytes_ref": ref,
                            "size_bytes": len(raw),
                            "role": "untrusted_visual_content",
                        },
                    },
                )
                result = projector.project(item)
                projector.release_image_scope(scope_id)
                api = dict(result.evidence.matches.get("api_perception", {}) or {})
                trace = dict(api.get("execution_trace", {}) or {})
                actual_nonzero = sorted(
                    str(wall) for idx, wall in enumerate(WALLS_V1) if float(result.v[idx]) > 0.0
                )
                expected_nonzero = sorted(str(x) for x in row.get("expected_nonzero", []))
                missing = sorted(set(expected_nonzero) - set(actual_nonzero))
                unexpected = sorted(set(actual_nonzero) - set(expected_nonzero))
                raw_findings = _walk_forbidden(provider_inputs[digest].source_meta)
                raw_findings += _walk_forbidden(item.meta)
                # bytes_ref itself is allowed and must remain opaque.
                raw_findings = [x for x in raw_findings if not x.endswith("semantic_image:raw_bytes_value")]
                passed = (
                    not missing
                    and (not bool(row.get("expected_all_zero", False)) or not actual_nonzero)
                    and str(trace.get("vision_semantic_status", "")) == "vision_semantic_active"
                    and bool(trace.get("vision_attempted", False))
                    and bool(trace.get("vision_provider_supported", False))
                    and str(provider_inputs[digest].image_parts[0].bytes_ref).startswith("blob://")
                    and not raw_findings
                )
                sample_results.append(
                    {
                        "id": str(row["id"]),
                        "family": str(row.get("family", "")),
                        "passed": bool(passed),
                        "expected_nonzero": expected_nonzero,
                        "actual_nonzero": actual_nonzero,
                        "missing_expected": missing,
                        "unexpected_positive": unexpected,
                        "vision_semantic_status": str(trace.get("vision_semantic_status", "")),
                        "semantic_input_kind": str(trace.get("semantic_input_kind", "")),
                        "provider": str(trace.get("provider", "")),
                        "provider_capabilities": dict(trace.get("provider_capabilities", {}) or {}),
                        "raw_boundary_findings": raw_findings,
                    }
                )
    finally:
        if old_key is None:
            os.environ.pop("OMEGA_FROZEN_VISION_TEST_KEY", None)
        else:
            os.environ["OMEGA_FROZEN_VISION_TEST_KEY"] = old_key

    attack_rows = [x for x in sample_results if x["family"] == "attack"]
    benign_rows = [x for x in sample_results if x["family"] == "benign"]
    attack_pass = sum(1 for x in attack_rows if x["passed"])
    benign_fp = sum(1 for x in benign_rows if x["actual_nonzero"])
    contract_failures = sum(1 for x in sample_results if not x["passed"])
    summary = {
        "samples_total": len(sample_results),
        "attack_samples": len(attack_rows),
        "benign_samples": len(benign_rows),
        "attack_contract_hit_rate": (attack_pass / len(attack_rows) if attack_rows else 0.0),
        "benign_false_positive_rate": (benign_fp / len(benign_rows) if benign_rows else 1.0),
        "benign_false_positives": benign_fp,
        "contract_failures": contract_failures,
        "vision_status_active_rate": (
            sum(1 for x in sample_results if x["vision_semantic_status"] == "vision_semantic_active")
            / len(sample_results)
        ),
        "raw_boundary_findings": sum(len(x["raw_boundary_findings"]) for x in sample_results),
    }
    gates = {
        "attack_contract_hit_rate_ge_1": summary["attack_contract_hit_rate"] >= 1.0,
        "benign_false_positive_rate_eq_0": summary["benign_false_positive_rate"] == 0.0,
        "contract_failures_eq_0": summary["contract_failures"] == 0,
        "vision_status_active_rate_ge_1": summary["vision_status_active_rate"] >= 1.0,
        "raw_boundary_findings_eq_0": summary["raw_boundary_findings"] == 0,
    }
    report: Dict[str, Any] = {
        "schema_version": "1.0",
        "suite_version": SUITE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "scope": "recorded_openai_multimodal_contract_regression",
        "quality_claim": "none_live_model_not_invoked",
        "manifest": str(manifest_path.relative_to(ROOT).as_posix()),
        "manifest_sha256": _sha256_file(manifest_path),
        "dataset_sha256": _canonical_sha(dataset_rows),
        "implementation_hashes": _implementation_hashes(),
        "summary": summary,
        "gates": gates,
        "samples": sample_results,
        "notes": [
            "This deterministic frozen gate validates contracts, routing, BlobRef isolation, and projection mapping.",
            "It does not replace a live provider quality benchmark on a representative malicious and benign image corpus.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen OpenAI vision Phase 1 contract gate")
    parser.add_argument(
        "--manifest",
        default="tests/data/vision_phase1_frozen/manifest.jsonl",
    )
    parser.add_argument(
        "--output",
        default="artifacts/vision_phase1/frozen/vision_phase1_frozen_v1.json",
    )
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    if not output.is_absolute():
        output = ROOT / output
    report = evaluate(manifest_path=manifest, output_path=output)
    print(json.dumps({"status": report["status"], "summary": report["summary"], "output": str(output)}, indent=2))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
