"""Public Omega Walls package surface.

Keep this module lightweight: importing a submodule such as ``omega.release``
must not require optional runtime dependencies used by the SDK/server path.
Heavy symbols are loaded lazily through ``__getattr__``.
"""

from __future__ import annotations

from omega.interfaces.contracts_v1 import K_V1, WALLS_V1
from omega.log_contract import AttributionItem, ErrorInfo, OmegaLogEvent
from omega.errors import (
    OmegaAPIError,
    OmegaConfigError,
    OmegaInitializationError,
    OmegaMissingDependencyError,
    OmegaRuntimeError,
    OmegaSDKError,
)
from omega.sdk_types import DetectionResult, GuardAction, GuardDecision, OmegaDetectionResult

__all__ = [
    "K_V1",
    "WALLS_V1",
    "OmegaWalls",
    "OmegaLogEvent",
    "AttributionItem",
    "ErrorInfo",
    "configure_omega_logging",
    "get_logger",
    "DetectionResult",
    "GuardDecision",
    "GuardAction",
    "OmegaDetectionResult",
    "OmegaSDKError",
    "OmegaMissingDependencyError",
    "OmegaConfigError",
    "OmegaAPIError",
    "OmegaInitializationError",
    "OmegaRuntimeError",
]


def __getattr__(name: str):
    if name == "OmegaWalls":
        from omega.sdk import OmegaWalls

        return OmegaWalls
    if name in {"configure_omega_logging", "get_logger"}:
        from omega.structured_logging import configure_omega_logging, get_logger

        return {"configure_omega_logging": configure_omega_logging, "get_logger": get_logger}[name]
    raise AttributeError(name)
