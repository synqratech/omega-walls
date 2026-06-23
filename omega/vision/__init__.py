"""Visual trust-boundary primitives for OCR and spatial analysis."""

from .contracts import OCRQualityPolicy, OCRQualitySummary, OCRSpan
from .spatial_policy import RegionPassPolicy, RegionPassDecision, decide_region_pass

__all__ = [
    "OCRQualityPolicy",
    "OCRQualitySummary",
    "OCRSpan",
    "RegionPassPolicy",
    "RegionPassDecision",
    "decide_region_pass",
]
