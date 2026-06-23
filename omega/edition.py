"""Edition boundary helpers for Omega Walls.

OSS code may expose extension points and compatibility shims, but paid
implementation modules live in the separate ``omega_walls_enterprise``
package and are not shipped in the community wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from types import ModuleType
from typing import Any


class EnterpriseFeatureUnavailable(RuntimeError):
    """Raised when Enterprise-only functionality is requested in Community."""


@dataclass(frozen=True)
class EditionStatus:
    community_package: str = "omega-walls"
    enterprise_package: str = "omega-walls-enterprise"
    enterprise_installed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "community_package": self.community_package,
            "enterprise_package": self.enterprise_package,
            "enterprise_installed": self.enterprise_installed,
        }


def is_enterprise_installed() -> bool:
    try:
        importlib.import_module("omega_walls_enterprise")
        return True
    except ModuleNotFoundError:
        return False


def edition_status() -> EditionStatus:
    return EditionStatus(enterprise_installed=is_enterprise_installed())


def require_enterprise(feature: str) -> None:
    raise EnterpriseFeatureUnavailable(
        f"{feature} is available in Omega Walls Enterprise only. "
        "Install the omega-walls-enterprise package or disable the enterprise feature."
    )


def import_enterprise_module(module_name: str, *, feature: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Only convert missing enterprise package/module into a product-level boundary error.
        if str(exc.name or "").split(".")[0] in {"omega_walls_enterprise", module_name.split(".")[0]}:
            raise EnterpriseFeatureUnavailable(
                f"{feature} requires omega-walls-enterprise; requested module {module_name!r} is not installed."
            ) from exc
        raise


def verify_runtime_license_if_available(config: Any) -> Any:
    """Run Enterprise runtime license verification when Enterprise is installed.

    Community runtime never imports legacy Enterprise namespace modules.  The
    optional Enterprise implementation must live in ``omega_walls_enterprise``.
    """
    try:
        module = import_enterprise_module(
            "omega_walls_enterprise.licensing.runtime_guard",
            feature="runtime license verification",
        )
    except EnterpriseFeatureUnavailable:
        return None
    verifier = getattr(module, "verify_runtime_license")
    return verifier(config)
