from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SignalSnapshot:
    scores: List[float]
    polarity: List[int]
    matches: Dict[str, Any]
