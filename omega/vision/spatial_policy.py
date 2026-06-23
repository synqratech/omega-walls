from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping


@dataclass(frozen=True)
class RegionPassPolicy:
    enabled: bool = False
    trigger_mode: str = "uncertain"  # zero | uncertain | always
    pressure_abs_max: float = 0.12
    confidence_max: float = 0.80
    max_tiles: int = 5
    overlap_ratio: float = 0.08
    include_center_crop: bool = True

    def __post_init__(self) -> None:
        mode = str(self.trigger_mode).strip().lower()
        if mode not in {"zero", "uncertain", "always"}:
            raise ValueError("image_region_pass.trigger_mode must be zero|uncertain|always")
        if not math.isfinite(float(self.pressure_abs_max)) or float(self.pressure_abs_max) < 0.0:
            raise ValueError("image_region_pass.pressure_abs_max must be finite and >= 0")
        if not math.isfinite(float(self.confidence_max)) or not 0.0 <= float(self.confidence_max) <= 1.0:
            raise ValueError("image_region_pass.confidence_max must be in [0,1]")
        if int(self.max_tiles) <= 0 or int(self.max_tiles) > 16:
            raise ValueError("image_region_pass.max_tiles must be in [1,16]")
        if not math.isfinite(float(self.overlap_ratio)) or not 0.0 <= float(self.overlap_ratio) <= 0.5:
            raise ValueError("image_region_pass.overlap_ratio must be in [0,0.5]")
        object.__setattr__(self, "trigger_mode", mode)


@dataclass(frozen=True)
class RegionPassDecision:
    run: bool
    reason: str
    max_abs_pressure: float
    confidence: float


def decide_region_pass(
    *,
    policy: RegionPassPolicy,
    pressure_signed: Mapping[str, float],
    confidence: float,
    has_image: bool,
    is_region_pass: bool,
) -> RegionPassDecision:
    pressures = [abs(float(value)) for value in pressure_signed.values()]
    max_abs = max(pressures) if pressures else 0.0
    conf = float(confidence)
    if not policy.enabled:
        return RegionPassDecision(False, "disabled", max_abs, conf)
    if not has_image:
        return RegionPassDecision(False, "no_image", max_abs, conf)
    if is_region_pass:
        return RegionPassDecision(False, "already_region_pass", max_abs, conf)
    if policy.trigger_mode == "always":
        return RegionPassDecision(True, "always", max_abs, conf)
    if policy.trigger_mode == "zero":
        return RegionPassDecision(max_abs <= 0.0, "zero_pressure" if max_abs <= 0.0 else "nonzero_pressure", max_abs, conf)
    uncertain = max_abs <= float(policy.pressure_abs_max) or conf <= float(policy.confidence_max)
    return RegionPassDecision(
        uncertain,
        "uncertain_low_pressure" if max_abs <= float(policy.pressure_abs_max) else ("uncertain_low_confidence" if uncertain else "confident"),
        max_abs,
        conf,
    )
