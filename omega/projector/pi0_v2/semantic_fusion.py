from __future__ import annotations

from typing import Any, Dict, List


def apply_semantic_fusion(
    self: Any,
    *,
    item_text: str,
    raw: List[float],
    raw_before_semantic: List[float],
    polarity: List[int],
) -> Dict[str, Any]:
    # Semantic fusion remains legacy-bound in this step; kept as dedicated seam for next wave.
    return {
        "raw": raw,
        "raw_before_semantic": raw_before_semantic,
        "polarity": polarity,
        "item_text": item_text,
    }
