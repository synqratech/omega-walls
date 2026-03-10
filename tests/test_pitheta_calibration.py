from __future__ import annotations

import numpy as np

from omega.pitheta.calibration import apply_temperature, build_temperature_payload


def test_temperature_payload_shapes_and_nll_improves_or_equal():
    rng = np.random.default_rng(41)
    ord_logits = rng.normal(size=(32, 4, 4)).astype(np.float32)
    pol_logits = rng.normal(size=(32, 4, 3)).astype(np.float32)
    ord_targets = rng.integers(low=0, high=4, size=(32, 4), endpoint=False, dtype=np.int64)
    pol_targets = rng.integers(low=0, high=3, size=(32, 4), endpoint=False, dtype=np.int64)

    payload = build_temperature_payload(
        ordinal_logits=ord_logits,
        ordinal_targets=ord_targets,
        polarity_logits=pol_logits,
        polarity_targets=pol_targets,
    )
    assert payload["schema_version"] == "pitheta_temperature_v1"
    assert len(payload["ordinal"]["temperatures"]) == 4
    assert len(payload["polarity"]["temperatures"]) == 4
    assert float(payload["ordinal"]["nll_after"]) <= float(payload["ordinal"]["nll_before"]) + 1e-9
    assert float(payload["polarity"]["nll_after"]) <= float(payload["polarity"]["nll_before"]) + 1e-9

    ord_scaled = apply_temperature(ord_logits, [1.0, 1.0, 1.0, 1.0])
    assert ord_scaled.shape == ord_logits.shape

