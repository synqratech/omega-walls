from __future__ import annotations

import json
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np

from scripts import eval_pitheta
from omega.pitheta.eval_gates import GateResult


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_ece_binary_basic_properties():
    probs = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    assert eval_pitheta._ece_binary(probs, labels) < 1e-9
    bad_probs = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    assert eval_pitheta._ece_binary(bad_probs, labels) > 0.5


def test_eval_report_contains_pitheta_calibration(monkeypatch):
    tmp = _mk_local_tmp("eval-pitheta-calibration")
    baseline_path = tmp / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "hard_negatives": {"fp": 0},
                "whitebox": {"base_detect_rate": 0.95, "bypass_rate": 0.20},
                "deepset": {"metrics": {"attack_off_rate": 0.20, "benign_off_rate": 0.0}},
                "omega": {"off_tau": 0.9},
                "pitheta_conversion": {"calibrated": True},
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    data_dir = tmp / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "holdout.jsonl").write_text("", encoding="utf-8")

    monkeypatch.setattr(
        eval_pitheta,
        "_run_eval",
        lambda **kwargs: {
            "hard_negatives": {"fp": 0},
            "whitebox": {"base_detect_rate": 0.96, "bypass_rate": 0.19},
            "deepset": {"metrics": {"attack_off_rate": 0.35, "benign_off_rate": 0.0, "attack_p95_sum_p": 1.0, "benign_p95_sum_p": 0.1}},
            "omega": {"off_tau": 0.9},
            "pitheta_conversion": {"calibrated": True, "pressure_map": [0.0, 0.25, 0.6, 1.0]},
            "_exit_code": 0,
            "_stdout": "{}",
            "_stderr": "",
        },
    )
    monkeypatch.setattr(
        eval_pitheta,
        "evaluate_pitheta_gates",
        lambda candidate, baseline: ("GO", [GateResult("PTG-001", "PASS", 0, "==0", "ok")]),
    )
    monkeypatch.setattr(
        eval_pitheta,
        "_compute_pitheta_calibration_report",
        lambda **kwargs: {
            "status": "ok",
            "samples": 10,
            "nll": {"ordinal": {"nll_before": 1.0, "nll_after": 0.8}},
            "ece_per_wall": {"override_instructions": 0.1},
        },
    )

    out_dir = tmp / "out"
    argv = [
        "eval_pitheta.py",
        "--checkpoint",
        str(tmp / "ckpt"),
        "--data-dir",
        data_dir.as_posix(),
        "--baseline-report",
        baseline_path.as_posix(),
        "--output-dir",
        out_dir.as_posix(),
    ]
    old_argv = list(sys.argv)
    try:
        sys.argv = argv
        rc = eval_pitheta.main()
    finally:
        sys.argv = old_argv
    assert rc == 0
    report = json.loads((out_dir / "eval_report.json").read_text(encoding="utf-8"))
    assert "pitheta_calibration" in report
    assert report["pitheta_calibration"]["status"] == "ok"

