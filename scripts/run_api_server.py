from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.api.server import create_app
from omega.config.loader import load_resolved_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Omega attachment scan HTTP API server.")
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--proxy-headers", dest="proxy_headers", action="store_true")
    parser.add_argument("--no-proxy-headers", dest="proxy_headers", action="store_false")
    parser.add_argument("--forwarded-allow-ips", default="*")
    parser.set_defaults(proxy_headers=True)
    args = parser.parse_args()

    snapshot = load_resolved_config(profile=args.profile)
    cfg = snapshot.resolved
    api_cfg = cfg.get("api", {}) or {}
    enabled = bool(api_cfg.get("enabled", True))
    if not enabled:
        raise SystemExit("api.enabled=false in config; refusing to start server")

    host = str(args.host or api_cfg.get("host", "127.0.0.1"))
    port = int(args.port if args.port is not None else api_cfg.get("port", 8080))
    app = create_app(resolved_config=cfg, profile=args.profile)

    import uvicorn

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=str(args.log_level),
        reload=bool(args.reload),
        proxy_headers=bool(args.proxy_headers),
        forwarded_allow_ips=str(args.forwarded_allow_ips),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
