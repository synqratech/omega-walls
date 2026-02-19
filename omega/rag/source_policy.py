"""Centralized trust policy for mapping source types to trust labels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


_TRUST_ALIAS = {
    "trusted": "trusted",
    "semi": "semi",
    "semi_trusted": "semi",
    "untrusted": "untrusted",
}


def _normalize_trust(value: str) -> str:
    norm = str(value).strip().lower()
    if norm not in _TRUST_ALIAS:
        raise ValueError(f"Unsupported trust level '{value}'")
    return _TRUST_ALIAS[norm]


@dataclass(frozen=True)
class SourceTrustPolicy:
    default_trust: str = "untrusted"
    source_type_to_trust: Dict[str, str] = field(default_factory=dict)
    source_prefix_to_trust: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Dict) -> "SourceTrustPolicy":
        raw = config.get("source_policy", {})
        source_type_to_trust = {
            str(k).strip().lower(): _normalize_trust(v)
            for k, v in (raw.get("source_type_to_trust", {}) or {}).items()
        }
        source_prefix_to_trust = {
            str(k): _normalize_trust(v)
            for k, v in (raw.get("source_prefix_to_trust", {}) or {}).items()
        }
        return cls(
            default_trust=_normalize_trust(raw.get("default_trust", "untrusted")),
            source_type_to_trust=source_type_to_trust,
            source_prefix_to_trust=source_prefix_to_trust,
        )

    def trust_for(self, source_type: str, source_id: str) -> str:
        sid = str(source_id)
        for prefix, trust in self.source_prefix_to_trust.items():
            if sid.startswith(prefix):
                return trust
        return self.source_type_to_trust.get(str(source_type).strip().lower(), self.default_trust)
