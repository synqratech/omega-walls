"""Community incident route shim.

Enterprise installs ``omega_walls_enterprise.incidents.routes``. Without it,
the API keeps a disabled/404 route surface and does not ship incident export
or replay implementation code.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from omega.edition import EnterpriseFeatureUnavailable, import_enterprise_module

try:
    _impl = import_enterprise_module("omega_walls_enterprise.incidents.routes", feature="incident export/replay routes")
except EnterpriseFeatureUnavailable:
    _impl = None

if _impl is not None:
    globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__") or k in {"__all__", "__doc__"}})
    __all__ = list(getattr(_impl, "__all__", [k for k in globals() if not k.startswith("_")]))
else:
    def build_incidents_router(runtime: Any) -> APIRouter:
        router = APIRouter(prefix="/v1", tags=["incidents"])

        @router.get("/incidents/health")
        async def incident_export_health() -> dict[str, Any]:
            return {
                "incident_export": {"enabled": False, "edition": "community"},
                "incident_replay": {"enabled": False, "edition": "community"},
            }

        @router.get("/incidents")
        async def incidents_disabled() -> None:
            raise HTTPException(status_code=404, detail="incident_export_enterprise_only")

        return router

    __all__ = ["build_incidents_router"]
