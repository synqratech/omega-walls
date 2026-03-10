"""PiTheta projector and hybrid combiner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem, ProjectionEvidence, ProjectionResult, WALLS_V1
from omega.projector.pitheta_inference import PiThetaInference, PiThetaInferenceConfig

WALLS = list(WALLS_V1)
INV_POLARITY_MAP = {0: -1, 1: 0, 2: 1}


def _zero_projection(item: ContentItem, reason: str) -> ProjectionResult:
    return ProjectionResult(
        doc_id=item.doc_id,
        v=np.zeros(4, dtype=float),
        evidence=ProjectionEvidence(
            polarity=[0, 0, 0, 0],
            debug_scores_raw=[0.0, 0.0, 0.0, 0.0],
            matches={"pitheta": {"active": False, "reason": reason}},
        ),
    )


@dataclass
class PiThetaProjector:
    config: Dict[str, Any]

    def __post_init__(self) -> None:
        projector_cfg = self.config.get("projector", {}) or {}
        pitheta_cfg = projector_cfg.get("pitheta", {}) or {}
        self.enabled_mode = str(pitheta_cfg.get("enabled", "auto")).lower()
        self.head_mode = str(pitheta_cfg.get("head_mode", "auto")).lower()
        self.conversion_mode = str(pitheta_cfg.get("conversion_mode", "expected")).lower()
        self.pressure_map = [float(x) for x in list(pitheta_cfg.get("pressure_map", [0.0, 0.25, 0.6, 1.0]))]
        self.require_calibration = bool(pitheta_cfg.get("require_calibration", True))
        self.calibration_file = str(pitheta_cfg.get("calibration_file", "temperature_scaling.json"))
        self.legacy_thresholds = dict(
            (pitheta_cfg.get("legacy", {}) or {}).get(
                "wall_thresholds",
                pitheta_cfg.get(
                    "wall_thresholds",
                    {
                        "override_instructions": 0.55,
                        "secret_exfiltration": 0.60,
                        "tool_or_action_abuse": 0.60,
                        "policy_evasion": 0.55,
                    },
                ),
            )
            or {}
        )
        self._runtime = PiThetaInference(
            PiThetaInferenceConfig(
                checkpoint_dir=str(pitheta_cfg.get("checkpoint_dir", "artifacts/pitheta_export/latest")),
                base_model_path=str(pitheta_cfg.get("base_model_path", "deberta-v3-base")),
                max_length=int(pitheta_cfg.get("max_length", 256)),
                batch_size=int(pitheta_cfg.get("batch_size", 8)),
                device=str(pitheta_cfg.get("device", "auto")),
                local_files_only=bool(pitheta_cfg.get("local_files_only", True)),
                head_mode=self.head_mode,
                calibration_file=self.calibration_file,
                require_calibration=self.require_calibration,
            )
        )
        if self.enabled_mode == "true" and not self._runtime.ready:
            raise RuntimeError(self._runtime.runtime_error or "PiTheta runtime inactive")

    def ensure_semantic_active(self) -> bool:
        return False

    def ensure_pitheta_calibration(self) -> bool:
        if not self._runtime.ready:
            return False
        if self._runtime.head_version == "legacy_v1":
            return True
        return bool(self._runtime.calibration_active)

    def pitheta_conversion_status(self) -> Dict[str, Any]:
        return {
            "head_version": self._runtime.head_version,
            "conversion_mode": self.conversion_mode,
            "pressure_map": list(self.pressure_map),
            "calibrated": bool(self._runtime.calibration_active),
            "require_calibration": bool(self.require_calibration),
            "active": bool(self._runtime.ready),
            "error": self._runtime.runtime_error,
            "legacy_thresholds": dict(self.legacy_thresholds),
            "metadata": self._runtime.metadata,
        }

    def semantic_status(self) -> Dict[str, Any]:
        return {
            "enabled_mode": "n/a",
            "active": False,
            "attempted": False,
            "model_path": None,
            "error": None,
            "docs_total": 0,
            "docs_with_boost": 0,
            "docs_polarity_promoted": 0,
            "docs_guard_suppressed": 0,
            "docs_with_boost_rate": 0.0,
            "promoted_polarity_rate": 0.0,
            "guard_suppression_rate": 0.0,
            "pitheta_active": bool(self._runtime.ready),
            "pitheta_error": self._runtime.runtime_error,
            "pitheta_metadata": self._runtime.metadata,
        }

    def _convert_ordinal(self, ordinal_prob: np.ndarray, polarity: List[int]) -> List[float]:
        values: List[float] = []
        for i in range(4):
            if int(polarity[i]) != 1:
                values.append(0.0)
                continue
            probs = ordinal_prob[i]
            if self.conversion_mode == "argmax":
                ord_idx = int(np.argmax(probs))
                values.append(float(self.pressure_map[ord_idx]))
            else:
                expected = 0.0
                for level in range(4):
                    expected += float(probs[level]) * float(self.pressure_map[level])
                values.append(float(expected))
        return values

    def project(self, item: ContentItem) -> ProjectionResult:
        if not self._runtime.ready:
            if self.enabled_mode == "true":
                raise RuntimeError(self._runtime.runtime_error or "PiTheta runtime inactive")
            return _zero_projection(item, reason=self._runtime.runtime_error or "pitheta_inactive")

        outputs = self._runtime.predict_outputs([item.text])
        head_version = str(outputs.get("head_version", "legacy_v1"))
        polarity_prob = outputs.get("polarity_prob")
        if not isinstance(polarity_prob, np.ndarray) or polarity_prob.shape[0] != 1:
            return _zero_projection(item, reason="invalid_polarity_shape")

        polarity_idx = np.argmax(polarity_prob[0], axis=-1).tolist()
        polarity = [int(INV_POLARITY_MAP[int(idx)]) for idx in polarity_idx]

        if head_version == "ordinal_v2":
            ord_prob = outputs.get("ordinal_prob")
            if not isinstance(ord_prob, np.ndarray) or ord_prob.shape[0] != 1:
                return _zero_projection(item, reason="invalid_ordinal_shape")
            raw = self._convert_ordinal(ord_prob[0], polarity=polarity)
            v = [float(x) for x in raw]
            matches = {
                "pitheta": {
                    "active": True,
                    "head_version": head_version,
                    "conversion_mode": self.conversion_mode,
                    "pressure_map": list(self.pressure_map),
                    "wall_prob": {WALLS[i]: float(v[i]) for i in range(4)},
                    "ordinal_prob": {WALLS[i]: [float(x) for x in ord_prob[0][i].tolist()] for i in range(4)},
                    "polarity_prob": {WALLS[i]: [float(x) for x in polarity_prob[0][i].tolist()] for i in range(4)},
                    "calibrated": bool(outputs.get("calibrated", False)),
                }
            }
        else:
            wall_prob = outputs.get("wall_prob")
            if not isinstance(wall_prob, np.ndarray) or wall_prob.shape[0] != 1:
                return _zero_projection(item, reason="invalid_wall_shape")
            raw = wall_prob[0].tolist()
            v = []
            for i, wall in enumerate(WALLS):
                threshold = float(self.legacy_thresholds.get(wall, 0.5))
                score = float(raw[i])
                pol = int(polarity[i])
                active = score >= threshold and pol == 1
                v.append(score if active else 0.0)
            matches = {
                "pitheta": {
                    "active": True,
                    "head_version": head_version,
                    "wall_prob": {WALLS[i]: float(raw[i]) for i in range(4)},
                    "polarity_prob": {WALLS[i]: [float(x) for x in polarity_prob[0][i].tolist()] for i in range(4)},
                    "thresholds": dict(self.legacy_thresholds),
                    "calibrated": bool(outputs.get("calibrated", False)),
                }
            }

        return ProjectionResult(
            doc_id=item.doc_id,
            v=np.array(v, dtype=float),
            evidence=ProjectionEvidence(
                polarity=polarity,
                debug_scores_raw=[float(x) for x in raw],
                matches=matches,
            ),
        )

    def fit(self, items: List[ContentItem], y: np.ndarray) -> None:  # pragma: no cover - API compatibility
        _ = (items, y)
        raise NotImplementedError(
            "PiThetaProjector runtime is inference-only. "
            "Use scripts/train_pitheta_lora.py for training."
        )


@dataclass
class HybridProjector:
    pi0_projector: Any
    pitheta_projector: PiThetaProjector

    def ensure_semantic_active(self) -> bool:
        return bool(getattr(self.pi0_projector, "ensure_semantic_active", lambda: False)())

    def ensure_pitheta_calibration(self) -> bool:
        return bool(getattr(self.pitheta_projector, "ensure_pitheta_calibration", lambda: False)())

    def pitheta_conversion_status(self) -> Dict[str, Any]:
        return dict(getattr(self.pitheta_projector, "pitheta_conversion_status", lambda: {})())

    def semantic_status(self) -> Dict[str, Any]:
        base = dict(getattr(self.pi0_projector, "semantic_status", lambda: {})())
        base["pitheta_active"] = bool(self.pitheta_projector._runtime.ready)
        base["pitheta_error"] = self.pitheta_projector._runtime.runtime_error
        return base

    def project(self, item: ContentItem) -> ProjectionResult:
        p0 = self.pi0_projector.project(item)
        pt = self.pitheta_projector.project(item)
        v = np.maximum(p0.v, pt.v)
        polarity: List[int] = []
        raw: List[float] = []
        for i in range(4):
            if float(pt.v[i]) > float(p0.v[i]):
                polarity.append(int(pt.evidence.polarity[i]))
                raw.append(float(pt.evidence.debug_scores_raw[i]))
            else:
                polarity.append(int(p0.evidence.polarity[i]))
                raw.append(float(p0.evidence.debug_scores_raw[i]))
        matches = {
            "hybrid": {
                "mode": "max",
                "walls": WALLS,
            },
            "pi0": dict(p0.evidence.matches),
            "pitheta": dict(pt.evidence.matches),
        }
        return ProjectionResult(
            doc_id=item.doc_id,
            v=v,
            evidence=ProjectionEvidence(
                polarity=polarity,
                debug_scores_raw=raw,
                matches=matches,
            ),
        )

    def __getattr__(self, name: str):
        # Keep compatibility with utilities that expect pi0-specific attributes (e.g. whitebox heuristics).
        return getattr(self.pi0_projector, name)

