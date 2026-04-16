"""CLI entrypoint for Omega attachment scan API server."""

from __future__ import annotations

import argparse

from omega.config.loader import load_resolved_config


def _build_missing_api_deps_message(*, missing_dep: str) -> str:
    dep = str(missing_dep or "unknown")
    return (
        f"Missing optional dependency for omega-walls-api: {dep}. "
        'Install with: pip install "omega-walls[api]"'
    )


def _create_api_app(*, cfg: dict, profile: str):
    from omega.api import create_app

    return create_app(resolved_config=cfg, profile=profile)


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
    try:
        app = _create_api_app(cfg=cfg, profile=args.profile)
    except ModuleNotFoundError as exc:
        dep = str(getattr(exc, "name", "") or "unknown")
        raise SystemExit(_build_missing_api_deps_message(missing_dep=dep)) from exc

    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise SystemExit(
            "uvicorn is required for omega-walls-api. Install with: pip install omega-walls[api]"
        ) from exc

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
