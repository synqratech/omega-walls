from __future__ import annotations

from scripts.run_pitheta_precloud_gate import _check_ece_non_worse, _check_nll_improvement


def test_check_nll_improvement():
    ok, details = _check_nll_improvement(
        {
            "nll": {
                "ordinal": {"nll_before": 1.2, "nll_after": 1.0},
                "polarity": {"nll_before": 0.8, "nll_after": 0.7},
            }
        }
    )
    assert ok is True
    assert details["ordinal"] is True
    assert details["polarity"] is True


def test_check_ece_non_worse():
    ok, details = _check_ece_non_worse(
        {"ece_per_wall": {"override_instructions": 0.08, "secret_exfiltration": 0.10}},
        {"ece_per_wall": {"override_instructions": 0.10, "secret_exfiltration": 0.10}},
    )
    assert ok is True
    assert details["override_instructions"] is True

