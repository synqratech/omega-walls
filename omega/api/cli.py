"""CLI entrypoint for Omega attachment scan API server."""

from __future__ import annotations

import argparse
import os
from typing import Sequence

from omega.config.loader import load_resolved_config


_PRODUCTION_PROFILE_ENVS = {
    "prod",
    "production",
    "prod_api",
    "prod_vision",
    "prod_vision_local_ocr",
    "prod_enterprise",
    "prod_vision_enterprise",
}


def _build_missing_api_deps_message(*, missing_dep: str) -> str:
    dep = str(missing_dep or "unknown")
    return (
        f"Missing optional dependency for omega-walls-api: {dep}. "
        'Install with: pip install "omega-walls[api]"'
    )


def _create_api_app(*, cfg: dict, profile: str):
    from omega.api import create_app

    return create_app(resolved_config=cfg, profile=profile)


def _resolve_profile(cli_profile: str | None) -> str:
    """Resolve the API profile with explicit CLI precedence and a secure default."""
    if cli_profile is not None and str(cli_profile).strip():
        return str(cli_profile).strip()
    env_profile = str(os.environ.get("OMEGA_PROFILE", "")).strip()
    return env_profile or "prod"


def _normalize_forwarded_allow_ips(values: Sequence[str]) -> str:
    normalized = [str(value).strip() for value in values if str(value).strip()]
    if not normalized:
        raise SystemExit("proxy headers require explicit api.security.trusted_proxy_cidrs")
    if any(value == "*" for value in normalized):
        raise SystemExit('refusing wildcard proxy trust: forwarded_allow_ips="*"')
    return ",".join(normalized)


def _resolve_proxy_runtime(*, cfg: dict, requested: bool | None, explicit_allow_ips: str | None) -> tuple[bool, str]:
    """Keep Uvicorn from rewriting peer/scheme unless explicitly and narrowly enabled.

    Omega validates raw proxy headers itself against ``trusted_proxy_cidrs``.  The
    default therefore leaves Uvicorn proxy-header processing disabled so the ASGI
    peer cannot be replaced before the trust-boundary check runs.
    """
    security = ((cfg.get("api", {}) or {}).get("security", {}) or {})
    proxy_headers = bool(requested) if requested is not None else False
    trusted = [str(x).strip() for x in list(security.get("trusted_proxy_cidrs", [])) if str(x).strip()]
    allow_ips = str(explicit_allow_ips or "").strip()
    if proxy_headers:
        profile_env = str(((cfg.get("profiles", {}) or {}).get("env", ""))).strip().lower()
        if profile_env in _PRODUCTION_PROFILE_ENVS:
            raise SystemExit(
                "production API refuses Uvicorn proxy-header rewriting; "
                "Omega validates X-Forwarded-Proto against trusted_proxy_cidrs itself"
            )
        if allow_ips == "*":
            raise SystemExit('refusing wildcard proxy trust: forwarded_allow_ips="*"')
        allow_ips = allow_ips or _normalize_forwarded_allow_ips(trusted)
    else:
        # Uvicorn ignores this value when proxy_headers=False, but keep it narrow.
        allow_ips = allow_ips or _normalize_forwarded_allow_ips(trusted or ["127.0.0.1/32", "::1/128"])
    return proxy_headers, allow_ips


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Omega attachment scan HTTP API server.")
    parser.add_argument("--profile", default=None, help="Config profile (default: OMEGA_PROFILE or prod)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--proxy-headers", dest="proxy_headers", action="store_true")
    parser.add_argument("--no-proxy-headers", dest="proxy_headers", action="store_false")
    parser.add_argument("--forwarded-allow-ips", default=None)
    parser.set_defaults(proxy_headers=None)
    args = parser.parse_args()

    profile = _resolve_profile(args.profile)
    snapshot = load_resolved_config(profile=profile)
    cfg = snapshot.resolved
    api_cfg = cfg.get("api", {}) or {}
    enabled = bool(api_cfg.get("enabled", True))
    if not enabled:
        raise SystemExit("api.enabled=false in config; refusing to start server")

    host = str(args.host or api_cfg.get("host", "127.0.0.1"))
    port = int(args.port if args.port is not None else api_cfg.get("port", 8080))
    proxy_headers, forwarded_allow_ips = _resolve_proxy_runtime(
        cfg=cfg,
        requested=args.proxy_headers,
        explicit_allow_ips=args.forwarded_allow_ips,
    )

    try:
        app = _create_api_app(cfg=cfg, profile=profile)
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
        proxy_headers=proxy_headers,
        forwarded_allow_ips=forwarded_allow_ips,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
