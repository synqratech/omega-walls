from __future__ import annotations

import pytest

from omega.config.loader import load_resolved_config
from omega.pitheta.dataset_builder import _validate_license_policy


def test_pitheta_registry_config_present_and_weights_positive():
    cfg = load_resolved_config(profile="dev").resolved
    reg = cfg.get("pitheta_dataset_registry", {})
    assert isinstance(reg, dict)
    datasets = reg.get("datasets", [])
    assert isinstance(datasets, list) and datasets
    weights = [float(d.get("sampling_weight", 0.0)) for d in datasets]
    assert all(w > 0 for w in weights)
    assert abs(sum(weights) - 1.0) < 1e-6


def test_license_policy_rejects_non_permissive():
    with pytest.raises(ValueError):
        _validate_license_policy(
            {
                "dataset_id": "x",
                "license_policy": "non_permissive",
                "allowed_for_train": True,
            }
        )
