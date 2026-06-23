#!/usr/bin/env python3
"""Export the generated Omega API OpenAPI document including attachment vision schemas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.api import server as api_server  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export attachment scan OpenAPI contract")
    parser.add_argument(
        "--output",
        default="enterprise/integrations/openapi/attachment_scan_openapi_v1.json",
    )
    parser.add_argument("--profile", default="dev")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    snapshot = api_server.load_resolved_config(profile=str(args.profile))
    app = api_server.create_app(resolved_config=snapshot.resolved, profile=str(args.profile))
    spec = app.openapi()
    if "/v1/scan/attachment" not in spec.get("paths", {}):
        raise RuntimeError("attachment scan route missing from generated OpenAPI")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
