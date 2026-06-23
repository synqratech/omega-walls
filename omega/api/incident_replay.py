"""Community compatibility surface for Enterprise incident replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from omega.edition import EnterpriseFeatureUnavailable, import_enterprise_module

try:
    _impl = import_enterprise_module("omega_walls_enterprise.incidents.replay", feature="incident replay")
except EnterpriseFeatureUnavailable:
    _impl = None

if _impl is not None:
    globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__") or k in {"__all__", "__doc__"}})
    __all__ = list(getattr(_impl, "__all__", [k for k in globals() if not k.startswith("_")]))
else:
    @dataclass(frozen=True)
    class IncidentReplayConfig:
        enabled: bool = False
        contract_version: str = "1.0.0"
        store_path: str = ""
        package_storage_path: str = ""
        encryption_key_env: str = ""
        download_ttl_hours: int = 0
        job_ttl_hours: int = 0
        max_steps: int = 0
        worker_max_concurrent_jobs: int = 0
        required_scope_read: str = "incidents:replay:read"
        required_scope_raw: str = "incidents:replay:raw"

        @classmethod
        def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "IncidentReplayConfig":
            data = dict(cfg or {})
            if bool(data.get("enabled", False)):
                raise EnterpriseFeatureUnavailable("incident replay is available in Omega Walls Enterprise only")
            return cls(enabled=False)

    class _EnterpriseOnlyStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise EnterpriseFeatureUnavailable("incident replay is available in Omega Walls Enterprise only")

    IncidentReplayStore = _EnterpriseOnlyStore
    IncidentReplayJobManager = _EnterpriseOnlyStore
    ReplayGenerateRequest = Any  # type: ignore
    ReplayGenerateResponse = Any  # type: ignore
    ReplayJobResponse = Any  # type: ignore

    __all__ = [
        "IncidentReplayConfig",
        "IncidentReplayStore",
        "IncidentReplayJobManager",
        "ReplayGenerateRequest",
        "ReplayGenerateResponse",
        "ReplayJobResponse",
    ]
