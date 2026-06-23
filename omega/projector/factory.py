"""Projector factory for pi0/pitheta/hybrid/hybrid_api modes."""

from __future__ import annotations

import logging
from typing import Any, Dict

from omega.projector.api_hybrid_projector import APIPerceptionProjector, HybridAPIProjector
from omega.projector.pi0_intent_v2 import Pi0IntentAwareV2
from omega.projector.pitheta_projector import HybridProjector, PiThetaProjector

LOGGER = logging.getLogger(__name__)


def build_projector(config: Dict[str, Any]):
    projector_cfg = config.get("projector", {}) or {}
    mode = str(projector_cfg.get("mode", "pi0")).strip().lower()
    fallback_to_pi0 = bool(projector_cfg.get("fallback_to_pi0", True))
    api_cfg = projector_cfg.get("api_perception", {}) if isinstance(projector_cfg.get("api_perception", {}), dict) else {}
    semantic_mode = api_cfg.get("semantic_mode")
    if semantic_mode is not None and mode != "hybrid_api":
        LOGGER.warning(
            "projector.api_perception.semantic_mode=%r is set but projector.mode=%s; semantic_mode applies only to hybrid_api",
            semantic_mode,
            mode,
        )

    pi0 = Pi0IntentAwareV2(config)
    if mode == "pi0":
        return pi0

    if mode == "hybrid_api":
        try:
            api_proj = APIPerceptionProjector(config)
        except Exception as exc:
            if fallback_to_pi0:
                LOGGER.warning("failed to initialize api perception projector; fallback to pi0: %s", exc)
                return pi0
            raise
        return HybridAPIProjector(pi0_projector=pi0, api_projector=api_proj)

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
