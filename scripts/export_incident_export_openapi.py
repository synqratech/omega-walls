from __future__ import annotations

import argparse
import json
from pathlib import Path

from omega.api.server import create_app
from omega.config.loader import load_resolved_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OpenAPI spec for Incident Export API")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--out", default="enterprise/integrations/openapi/incident_export_openapi_v1.json")
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=str(args.profile))
    app = create_app(resolved_config=snapshot.resolved, profile=str(args.profile))
    spec = app.openapi()

    out_path = Path(str(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "out": str(out_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
