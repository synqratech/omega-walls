from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.api import server as api_server
from omega.config.loader import load_resolved_config
from omega.env_file import load_repo_env_file


def _load_key() -> None:
    load_repo_env_file()


def _build_runtime(
    *,
    profile: str,
    ocr_enabled: str,
    ocr_provider: str,
    pi0_semantic_enabled: str,
    pi0_semantic_device: str,
) -> Any:
    overrides: Dict[str, Any] = {
        "runtime": {"guard_mode": "enforce"},
        "projector": {
            "mode": "hybrid_api",
            "api_perception": {
                "enabled": "true",
                "provider": "openai",
                "provider_options": {
                    "capabilities": {
                        "text": True,
                        "image": True,
                    }
                },
            },
        },
        "pi0": {
            "semantic": {
                "enabled": str(pi0_semantic_enabled),
                "device": str(pi0_semantic_device),
            }
        },
        "retriever": {
            "sqlite_fts": {
                "attachments": {
                    "ocr": {
                        "enabled": str(ocr_enabled),
                        "provider": str(ocr_provider),
                    }
                }
            }
        },
    }
    snapshot = load_resolved_config(profile=profile, cli_overrides=overrides)
    return api_server._make_runtime(snapshot.resolved)


def _mime_for(path: Path) -> str:
    return {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
    }.get(str(path.suffix).strip().lower(), "application/octet-stream")


def _scan_case(runtime: Any, path: Path, *, request_tag: str) -> Dict[str, Any]:
    payload = {
        "tenant_id": "image-diagnostic",
        "request_id": f"{request_tag}-{path.stem}",
        "use_extracted_text": False,
        "file_bytes": path.read_bytes(),
        "filename": path.name,
        "mime": _mime_for(path),
    }
    out = api_server._scan_request(runtime, payload)
    trace = out.get("policy_trace", {}) if isinstance(out.get("policy_trace"), dict) else {}
    chunk_pipeline = trace.get("chunk_pipeline", {}) if isinstance(trace.get("chunk_pipeline"), dict) else {}
    return {
        "verdict": str(out.get("verdict", "")),
        "risk_score": int(out.get("risk_score", 0)),
        "reasons": [str(x) for x in list(out.get("reasons", []) or [])],
        "semantic_status": str(trace.get("semantic_failure_status", "")),
        "vision_semantic_status": str(trace.get("vision_semantic_status", "")),
        "ocr_status": str(trace.get("ocr_status", "none")),
        "ocr_provider": str(trace.get("ocr_provider", "")),
        "raw_modality_verdict": {
            "ocr": str(trace.get("ocr_max_chunk_score_band", "allow")),
            "vision": str(trace.get("vision_max_chunk_score_band", "allow")),
        },
        "active_walls": {
            "ocr": [str(x) for x in list(trace.get("ocr_active_walls", []) or [])],
            "vision": [str(x) for x in list(trace.get("vision_active_walls", []) or [])],
        },
        "active_modalities": [str(x) for x in list(trace.get("active_modalities", []) or [])],
        "modality_positive_chunk_counts": dict(trace.get("modality_positive_chunk_counts", {}) or {}),
        "modality_wall_max": dict(trace.get("modality_wall_max", {}) or {}),
        "ocr_vision_agreement": bool(trace.get("ocr_vision_agreement", False)),
        "adjudication": {
            "gate_applied": bool(trace.get("ocr_gate_applied", False)),
            "gate_reason": str(trace.get("ocr_gate_reason", "none")),
            "status": str(trace.get("ocr_adjudication_status", "not_needed")),
            "result": str(trace.get("ocr_adjudication_result", "none")),
        },
        "top_chunks": [dict(x) for x in list(chunk_pipeline.get("top_chunks", []) or [])[:5]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose one image attachment across vision-only and vision+OCR paths."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--ocr-provider", default="rapidocr")
    parser.add_argument("--pi0-semantic-enabled", default="true")
    parser.add_argument("--pi0-semantic-device", default="auto")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    _load_key()
    image_path = Path(str(args.image))
    if not image_path.is_absolute():
        image_path = (ROOT / image_path).resolve()
    if not image_path.exists():
        raise SystemExit(f"image not found: {image_path}")
    if not str(os.environ.get("OPENAI_API_KEY", "")).strip():
        raise SystemExit("OPENAI_API_KEY is not set; populate .env or environment first")

    vision_only_runtime = _build_runtime(
        profile=str(args.profile),
        ocr_enabled="false",
        ocr_provider=str(args.ocr_provider),
        pi0_semantic_enabled=str(args.pi0_semantic_enabled),
        pi0_semantic_device=str(args.pi0_semantic_device),
    )
    vision_plus_ocr_runtime = _build_runtime(
        profile=str(args.profile),
        ocr_enabled="true",
        ocr_provider=str(args.ocr_provider),
        pi0_semantic_enabled=str(args.pi0_semantic_enabled),
        pi0_semantic_device=str(args.pi0_semantic_device),
    )

    payload = {
        "image": str(image_path),
        "profile": str(args.profile),
        "ocr_provider": str(args.ocr_provider),
        "pi0_semantic_enabled": str(args.pi0_semantic_enabled),
        "pi0_semantic_device": str(args.pi0_semantic_device),
        "vision_only": _scan_case(vision_only_runtime, image_path, request_tag="vision-only"),
        "vision_plus_ocr": _scan_case(vision_plus_ocr_runtime, image_path, request_tag="vision-ocr"),
    }
    out_path = str(args.out or "").strip()
    if out_path:
        target = Path(out_path)
        if not target.is_absolute():
            target = (ROOT / target).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
