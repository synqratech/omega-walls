"""Projector factory for pi0/pitheta/hybrid modes."""

from __future__ import annotations

import logging
from typing import Any, Dict

from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from omega.projector.pitheta_projector import HybridProjector, PiThetaProjector

LOGGER = logging.getLogger(__name__)


def build_projector(config: Dict[str, Any]):
    projector_cfg = config.get("projector", {}) or {}
    mode = str(projector_cfg.get("mode", "pi0")).strip().lower()
    fallback_to_pi0 = bool(projector_cfg.get("fallback_to_pi0", True))

    pi0 = Pi0IntentAwareV2(config)
    if mode == "pi0":
        return pi0

    try:
        pitheta = PiThetaProjector(config)
    except Exception as exc:
        if fallback_to_pi0:
            LOGGER.warning("failed to initialize pitheta projector; fallback to pi0: %s", exc)
            return pi0
        raise

    if mode == "pitheta":
        return pitheta
    if mode == "hybrid":
        return HybridProjector(pi0_projector=pi0, pitheta_projector=pitheta)

    if fallback_to_pi0:
        LOGGER.warning("unknown projector.mode=%s; fallback to pi0", mode)
        return pi0
    raise ValueError(f"unsupported projector.mode={mode}")
