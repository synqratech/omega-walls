from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from omega.interfaces.contracts_v1 import ContentItem
from omega.projector import pitheta_projector as pp


@dataclass
class _FakeRuntime:
    ready: bool = True
    runtime_error: str | None = None
    head_version: str = "ordinal_v2"
    calibration_active: bool = True

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"mock": True}

    def predict_outputs(self, texts):
        _ = texts
        return {
            "head_version": self.head_version,
            "calibrated": self.calibration_active,
            "ordinal_prob": np.array(
                [
                    [
                        [0.1, 0.2, 0.3, 0.4],  # override
                        [1.0, 0.0, 0.0, 0.0],  # secret
                        [0.0, 1.0, 0.0, 0.0],  # tool
                        [0.0, 0.0, 1.0, 0.0],  # evasion
                    ]
                ],
                dtype=np.float32,
            ),
            "wall_prob": np.zeros((1, 4), dtype=np.float32),
            "polarity_prob": np.array(
                [
                    [
                        [0.0, 0.0, 1.0],  # +1
                        [1.0, 0.0, 0.0],  # -1
                        [0.0, 1.0, 0.0],  # 0
                        [0.0, 0.0, 1.0],  # +1
                    ]
                ],
                dtype=np.float32,
            ),
        }


def test_ordinal_expected_conversion_respects_polarity_gate(monkeypatch):
    monkeypatch.setattr(pp, "PiThetaInference", lambda cfg: _FakeRuntime())
    projector = pp.PiThetaProjector(
        config={
            "projector": {
                "pitheta": {
                    "enabled": "true",
                    "head_mode": "ordinal",
                    "conversion_mode": "expected",
                    "pressure_map": [0.0, 0.25, 0.6, 1.0],
                }
            }
        }
    )
    out = projector.project(
        ContentItem(
            doc_id="d1",
            source_id="s1",
            source_type="other",
            trust="untrusted",
            text="x",
        )
    )
    # override active with expected value, secret/tool zeroed by polarity gate, evasion active.
    assert float(out.v[0]) > 0.0
    assert float(out.v[1]) == 0.0
    assert float(out.v[2]) == 0.0
    assert float(out.v[3]) > 0.0

