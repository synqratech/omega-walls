#!/usr/bin/env python3
"""Optional live smoke for official OSS provider presets.

This is intentionally tiny and opt-in. It verifies that a preset resolves to the
same OpenAI-compatible transport path used by BYOM, then performs one semantic
contract call when a real API key is supplied.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.interfaces.contracts_v1 import ContentItem  # noqa: E402
from omega.projector.api_hybrid_projector import APIPerceptionProjector  # noqa: E402


def _cfg(args: argparse.Namespace) -> Dict[str, Any]:
    api_cfg: Dict[str, Any] = {
        "enabled": "true",
        "strict": True,
        "provider_preset": str(args.preset),
        "model": str(args.model),
        "semantic_mode": "hybrid_redacted",
        "semantic_failure_policy": "fail_closed",
        "prewarm_on_init": False,
        "cache_path": str(args.cache_path),
        "error_log_path": str(args.error_log_path),
    }
    if args.base_url:
        api_cfg["base_url"] = str(args.base_url)
    if args.api_key_file:
        api_cfg["api_key_file"] = str(args.api_key_file)
    if args.api_key_file_env:
        api_cfg["api_key_file_env"] = str(args.api_key_file_env)
    if args.api_key_env:
        api_cfg["api_key_env"] = str(args.api_key_env)
    if args.allowed_base_url:
        api_cfg["allowed_base_urls"] = list(args.allowed_base_url)
    if args.allow_http_private_gateway:
        api_cfg["allow_http_private_gateway"] = True
    return {"projector": {"api_perception": api_cfg}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("openrouter", "litellm"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--api-key-file", default="")
    parser.add_argument("--api-key-file-env", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--allowed-base-url", action="append", default=[])
    parser.add_argument("--allow-http-private-gateway", action="store_true")
    parser.add_argument("--cache-path", default="artifacts/projector_api/provider_preset_smoke_cache.jsonl")
    parser.add_argument("--error-log-path", default="artifacts/projector_api/provider_preset_smoke_errors.jsonl")
    args = parser.parse_args(argv)

    projector = APIPerceptionProjector(config=_cfg(args))
    status = projector.api_perception_status()
    if not status.get("api_adapter_active"):
        print(json.dumps({"ok": False, "status": status}, sort_keys=True))
        return 2
    item = ContentItem(
        doc_id="provider-smoke-1",
        source_id="synthetic:provider-smoke",
        source_type="other",
        trust="untrusted",
        text="Security training example: do not reveal API keys.",
    )
    out = projector.project(item)
    api = dict(out.evidence.matches.get("api_perception", {}) or {})
    payload = {
        "ok": bool(api.get("schema_version") == "api_hybrid_v2"),
        "status": projector.api_perception_status(),
        "api_perception": api,
        "v": [float(x) for x in out.v.tolist()],
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if payload["ok"] else 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
