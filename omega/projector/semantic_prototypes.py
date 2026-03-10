"""Semantic prototype helpers for pi0 projector."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from omega.interfaces.contracts_v1 import WALLS_V1


@dataclass(frozen=True)
class SemanticPrototypePack:
    walls: Dict[str, List[str]]
    guards: Dict[str, List[str]]


def load_semantic_prototypes(config: dict) -> SemanticPrototypePack:
    pi0 = config.get("pi0", config)
    sem = pi0.get("semantic", {})
    raw = sem.get("prototypes", {})
    guards = raw.get("guards", {})

    wall_items: Dict[str, List[str]] = {}
    for wall in WALLS_V1:
        vals = ((raw.get(wall) or {}).get("positive") or [])
        wall_items[wall] = [str(v).strip() for v in vals if str(v).strip()]

    guard_items: Dict[str, List[str]] = {}
    for key in ("negation", "protect", "tutorial"):
        vals = guards.get(key, []) or []
        guard_items[key] = [str(v).strip() for v in vals if str(v).strip()]

    return SemanticPrototypePack(walls=wall_items, guards=guard_items)
