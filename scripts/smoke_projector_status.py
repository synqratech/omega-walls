from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.config.loader import load_resolved_config
from omega.interfaces.contracts_v1 import ContentItem
from omega.projector.factory import build_projector


def _default_api_status() -> Dict[str, Any]:
    return {
        "api_adapter_active": False,
        "api_adapter_error": "not_supported",
        "schema_valid": None,
        "model": None,
        "cache_hit_rate": 0.0,
    }


def _status_label(api_status: Dict[str, Any]) -> str:
    if not bool(api_status.get("api_adapter_active", False)):
        return "error"
    schema_valid = api_status.get("schema_valid", None)
    if schema_valid is False:
        return "calibrated(error-schema)"
    return "active"


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke-check projector status (including API-hybrid contract).")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--mode", choices=["pi0", "pitheta", "hybrid", "hybrid_api"], default="hybrid_api")
    parser.add_argument("--probe-text", default="Summarize secure coding practices for internal docs.")
    parser.add_argument("--no-probe", action="store_true")
    parser.add_argument("--api-model", default=None)
    parser.add_argument("--api-base-url", default=None)
    args = parser.parse_args()

    projector_override: Dict[str, Any] = {"mode": str(args.mode)}
    if str(args.mode) == "hybrid_api":
        projector_override = {
            "mode": "hybrid_api",
            "fallback_to_pi0": False,
            "api_perception": {
                "enabled": "true",
                **({"model": str(args.api_model)} if args.api_model else {}),
                **({"base_url": str(args.api_base_url)} if args.api_base_url else {}),
            },
        }

    try:
        snapshot = load_resolved_config(profile=str(args.profile), cli_overrides={"projector": projector_override})
        projector = build_projector(snapshot.resolved)
    except Exception as exc:  # noqa: BLE001
        payload = {
            "status": "error",
            "mode": str(args.mode),
            "api_adapter_active": False,
            "api_adapter_error": str(exc),
            "schema_valid": None,
            "model": args.api_model,
            "cache_hit_rate": 0.0,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1

    api_status_fn = getattr(projector, "api_perception_status", None)
    api_status = api_status_fn() if callable(api_status_fn) else _default_api_status()
    if (not args.no_probe) and str(args.mode) == "hybrid_api" and bool(api_status.get("api_adapter_active", False)):
        try:
            _ = projector.project(
                ContentItem(
                    doc_id="smoke_api_hybrid_001",
                    source_id="smoke:status",
                    source_type="other",
                    trust="untrusted",
                    text=str(args.probe_text),
                )
            )
        except Exception as exc:  # noqa: BLE001
            api_status["api_adapter_error"] = str(exc)
            api_status["schema_valid"] = False
        api_status = api_status_fn() if callable(api_status_fn) else api_status

    payload = {
        "status": _status_label(api_status),
        "mode": str(args.mode),
        "api_adapter_active": bool(api_status.get("api_adapter_active", False)),
        "api_adapter_error": api_status.get("api_adapter_error"),
        "schema_valid": api_status.get("schema_valid"),
        "model": api_status.get("model"),
        "cache_hit_rate": float(api_status.get("cache_hit_rate", 0.0)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] in {"active", "calibrated(error-schema)"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

