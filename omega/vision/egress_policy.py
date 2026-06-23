"""Tenant-scoped visual egress and data-residency policy.

The policy is evaluated for every provider candidate before raw image bytes are
resolved from the request-scoped BlobStore.  This keeps residency enforcement
at the last trusted boundary before external transmission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import re


@dataclass(frozen=True)
class VisualEgressDecision:
    allowed: bool
    reason: str
    tenant_id: str
    data_region: str
    provider_id: str
    provider_type: str
    provider_region: str
    external: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": bool(self.allowed),
            "reason": str(self.reason),
            "tenant_id": str(self.tenant_id),
            "data_region": str(self.data_region),
            "provider_id": str(self.provider_id),
            "provider_type": str(self.provider_type),
            "provider_region": str(self.provider_region),
            "external": bool(self.external),
        }


_REGION_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")


def _region(value: Any, *, field_name: str) -> str:
    normalized = str(value or "").strip().lower()
    if not normalized or not _REGION_RE.fullmatch(normalized):
        raise ValueError(f"{field_name} must be a lowercase region token")
    return normalized


class VisualEgressPolicy:
    """Resolve tenant/provider compatibility without inspecting media bytes."""

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        raw = dict(cfg or {})
        self.enabled = bool(raw.get("enabled", False))
        self.default_action = (
            str(raw.get("default_action", "allow")).strip().lower() or "allow"
        )
        if self.default_action not in {"allow", "deny"}:
            raise ValueError("visual_egress.default_action must be allow|deny")
        providers = raw.get("providers", {}) or {}
        tenants = raw.get("tenants", {}) or {}
        if not isinstance(providers, Mapping) or not isinstance(tenants, Mapping):
            raise ValueError("visual_egress.providers and tenants must be mappings")
        self.providers = {
            str(k): dict(v or {})
            for k, v in providers.items()
            if isinstance(v, Mapping)
        }
        self.tenants = {
            str(k): dict(v or {}) for k, v in tenants.items() if isinstance(v, Mapping)
        }
        for provider_id, row in self.providers.items():
            if "external" in row and type(row.get("external")) is not bool:
                raise ValueError(
                    f"visual_egress.providers.{provider_id}.external must be boolean"
                )
            _region(
                row.get("region", "global"),
                field_name=f"visual_egress.providers.{provider_id}.region",
            )
        for tenant_id, row in self.tenants.items():
            for key in (
                "allow_external",
                "require_region_match",
                "require_data_region",
            ):
                if key in row and type(row.get(key)) is not bool:
                    raise ValueError(
                        f"visual_egress.tenants.{tenant_id}.{key} must be boolean"
                    )
            for key in ("allowed_providers", "allowed_regions"):
                if key in row and not isinstance(row.get(key), (list, tuple)):
                    raise ValueError(
                        f"visual_egress.tenants.{tenant_id}.{key} must be a list"
                    )

    def _provider_record(self, provider_id: str, provider_type: str) -> dict[str, Any]:
        row = dict(
            self.providers.get(str(provider_id), {})
            or self.providers.get(str(provider_type), {})
            or {}
        )
        external_default = str(provider_type).strip().lower() not in {
            "local_vision",
            "local",
        }
        row.setdefault("external", external_default)
        row.setdefault("region", "local" if not row["external"] else "global")
        return row

    def _tenant_record(self, tenant_id: str) -> dict[str, Any]:
        return dict(
            self.tenants.get(str(tenant_id), {}) or self.tenants.get("*", {}) or {}
        )

    def decide(
        self,
        *,
        tenant_id: str,
        data_region: str,
        provider_id: str,
        provider_type: str,
    ) -> VisualEgressDecision:
        tenant = str(tenant_id or "default")
        requested_region = _region(
            data_region or "unspecified", field_name="data_region"
        )
        provider = self._provider_record(provider_id, provider_type)
        provider_region = _region(
            provider.get("region", "global"), field_name="provider_region"
        )
        external = bool(provider.get("external", True))
        if not self.enabled:
            return VisualEgressDecision(
                allowed=True,
                reason="policy_disabled",
                tenant_id=tenant,
                data_region=requested_region,
                provider_id=str(provider_id),
                provider_type=str(provider_type),
                provider_region=provider_region,
                external=external,
            )

        rule = self._tenant_record(tenant)
        allow_external = bool(
            rule.get("allow_external", self.default_action == "allow")
        )
        allowed_providers = {
            str(x) for x in list(rule.get("allowed_providers", []) or []) if str(x)
        }
        allowed_regions = {
            str(x).strip().lower()
            for x in list(rule.get("allowed_regions", []) or [])
            if str(x).strip()
        }
        require_region_match = bool(rule.get("require_region_match", False))
        require_data_region = bool(rule.get("require_data_region", False))

        if require_data_region and requested_region == "unspecified":
            allowed, reason = False, "data_region_required"
        elif (
            allowed_providers
            and str(provider_id) not in allowed_providers
            and str(provider_type) not in allowed_providers
        ):
            allowed, reason = False, "provider_not_allowed_for_tenant"
        elif external and not allow_external:
            allowed, reason = False, "external_visual_egress_denied"
        elif allowed_regions and provider_region not in allowed_regions:
            allowed, reason = False, "provider_region_not_allowed"
        elif require_region_match and requested_region not in {
            "unspecified",
            provider_region,
        }:
            allowed, reason = False, "data_residency_region_mismatch"
        else:
            allowed, reason = True, "allowed"
        return VisualEgressDecision(
            allowed=allowed,
            reason=reason,
            tenant_id=tenant,
            data_region=requested_region,
            provider_id=str(provider_id),
            provider_type=str(provider_type),
            provider_region=provider_region,
            external=external,
        )
