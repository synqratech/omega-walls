from __future__ import annotations

from omega.pitheta.eval_gates import evaluate_pitheta_gates


def _base_payload() -> dict:
    return {
        "hard_negatives": {"fp": 0},
        "whitebox": {"base_detect_rate": 0.95, "bypass_rate": 0.20},
        "omega": {"off_tau": 0.90},
        "pitheta_conversion": {"calibrated": True},
        "deepset": {
            "metrics": {
                "attack_off_rate": 0.30,
                "benign_off_rate": 0.01,
                "coverage_wall_any_attack": 0.80,
                "attack_p95_sum_p": 0.85,
                "benign_p95_sum_p": 0.10,
            }
        },
    }


def test_pitheta_gates_go_path():
    baseline = _base_payload()
    candidate = {
        "hard_negatives": {"fp": 0},
        "whitebox": {"base_detect_rate": 0.97, "bypass_rate": 0.15},
        "omega": {"off_tau": 0.90},
        "pitheta_conversion": {"calibrated": True},
        "deepset": {
            "metrics": {
                "attack_off_rate": 0.42,
                "benign_off_rate": 0.01,
                "coverage_wall_any_attack": 0.85,
                "attack_p95_sum_p": 0.90,
                "benign_p95_sum_p": 0.12,
            }
        },
    }
    status, gates = evaluate_pitheta_gates(candidate, baseline)
    assert status == "GO"
    assert all(g.status == "PASS" for g in gates)


def test_pitheta_gates_fail_path():
    baseline = _base_payload()
    candidate = {
        "hard_negatives": {"fp": 1},
        "whitebox": {"base_detect_rate": 0.80, "bypass_rate": 0.40},
        "omega": {"off_tau": 0.90},
        "pitheta_conversion": {"calibrated": False},
        "deepset": {
            "metrics": {
                "attack_off_rate": 0.31,
                "benign_off_rate": 0.10,
                "coverage_wall_any_attack": 0.70,
                "attack_p95_sum_p": 0.10,
                "benign_p95_sum_p": 0.40,
            }
        },
    }
    status, gates = evaluate_pitheta_gates(candidate, baseline)
    assert status == "NO_GO"
    assert any(g.status == "FAIL" for g in gates)
