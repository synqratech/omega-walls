"""Community compatibility surface for Enterprise incident export.

The full incident database/export implementation is in
``omega_walls_enterprise.incidents.export``. Community keeps only disabled
placeholders plus a scan-record hook that returns ``None`` when export is not
installed/enabled.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from omega.edition import EnterpriseFeatureUnavailable, import_enterprise_module

try:
    _impl = import_enterprise_module("omega_walls_enterprise.incidents.export", feature="incident export")
except EnterpriseFeatureUnavailable:
    _impl = None

if _impl is not None:
    globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__") or k in {"__all__", "__doc__"}})
    __all__ = list(getattr(_impl, "__all__", [k for k in globals() if not k.startswith("_")]))
else:
    @dataclass(frozen=True)
    class IncidentExportConfig:
        enabled: bool = False
        contract_version: str = "1.0"
        store_path: str = ""
        key_store_path: str = ""
        retention_days: int = 0
        default_env: str = "dev"
        rate_limit_rpm: int = 0
        rate_limit_burst: int = 0
        cors_allowed_origins: list[str] | None = None

        @classmethod
        def from_cfg(cls, cfg: Mapping[str, Any] | None) -> "IncidentExportConfig":
            data = dict(cfg or {})
            if bool(data.get("enabled", False)):
                raise EnterpriseFeatureUnavailable("incident export is available in Omega Walls Enterprise only")
            return cls(enabled=False, cors_allowed_origins=[])

    class _EnterpriseOnlyStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise EnterpriseFeatureUnavailable("incident export is available in Omega Walls Enterprise only")

    IncidentExportStore = _EnterpriseOnlyStore
    IncidentApiKeyStore = _EnterpriseOnlyStore
    IncidentRateLimiter = _EnterpriseOnlyStore
    IncidentApiKeyRecord = Any  # type: ignore

    def key_fingerprint(value: str) -> str:
        return ""

    def build_incident_record_from_scan(*args: Any, **kwargs: Any) -> Optional[Dict[str, Any]]:
        return None

    __all__ = [
        "IncidentExportConfig",
        "IncidentExportStore",
        "IncidentApiKeyStore",
        "IncidentRateLimiter",
        "IncidentApiKeyRecord",
        "key_fingerprint",
        "build_incident_record_from_scan",
    ]
