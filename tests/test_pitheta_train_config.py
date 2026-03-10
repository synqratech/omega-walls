from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from omega.pitheta.training import load_train_config


def _mk_local_tmp(name: str) -> Path:
    root = Path("tests/_tmp")
    root.mkdir(parents=True, exist_ok=True)
    out = root / f"{name}-{uuid4().hex[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_train_config_valid():
    cfg = load_train_config("config/pitheta_train.yml")
    assert cfg.base_model
    assert cfg.batch_size > 0
    assert cfg.lora_r > 0
    assert cfg.use_ordinal is True
    assert cfg.fit_temperature is True
    assert cfg.calibration_source_mode in {"dataset", "gold_only", "blended"}
    assert cfg.temperature_split in {"dev", "holdout", "calibration"}
    assert cfg.calibration_gold_ratio >= 0.0
    assert cfg.calibration_weak_ratio >= 0.0


def test_train_config_invalid_fails():
    tmp = _mk_local_tmp("pitheta-train-cfg")
    path = tmp / "bad.yml"
    path.write_text(
        "pitheta_train:\n"
        "  base_model: microsoft/deberta-v3-base\n"
        "  task: ordinal_pressure_with_polarity\n"
        "  max_len: 0\n"
        "  batch_size: 16\n"
        "  lr: 0.0002\n"
        "  epochs: 3\n"
        "  loss_weights:\n"
        "    ordinal: 1.0\n"
        "    polarity: 0.3\n"
        "  labeling:\n"
        "    ordinal_bins: [0.1, 0.2, 0.3]\n"
        "  lora:\n"
        "    enabled: true\n"
        "    r: 16\n"
        "    alpha: 32\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _ = load_train_config(path.as_posix())


def test_train_config_invalid_blended_ratio_fails():
    tmp = _mk_local_tmp("pitheta-train-cfg-ratio")
    path = tmp / "bad_ratio.yml"
    path.write_text(
        "pitheta_train:\n"
        "  base_model: microsoft/deberta-v3-base\n"
        "  task: ordinal_pressure_with_polarity\n"
        "  max_len: 256\n"
        "  batch_size: 16\n"
        "  lr: 0.0002\n"
        "  epochs: 3\n"
        "  calibration:\n"
        "    source_mode: blended\n"
        "    gold_ratio: 0.8\n"
        "    weak_ratio: 0.3\n"
        "  loss_weights:\n"
        "    ordinal: 1.0\n"
        "    polarity: 0.3\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _ = load_train_config(path.as_posix())


def test_train_config_calibration_split_is_allowed():
    tmp = _mk_local_tmp("pitheta-train-cfg-cal-split")
    path = tmp / "ok_calibration.yml"
    path.write_text(
        "pitheta_train:\n"
        "  base_model: microsoft/deberta-v3-base\n"
        "  task: ordinal_pressure_with_polarity\n"
        "  max_len: 256\n"
        "  batch_size: 16\n"
        "  lr: 0.0002\n"
        "  epochs: 1\n"
        "  calibration:\n"
        "    fit_temperature: true\n"
        "    temperature_split: calibration\n"
        "    source_mode: gold_only\n"
        "  loss_weights:\n"
        "    ordinal: 1.0\n"
        "    polarity: 0.3\n",
        encoding="utf-8",
    )
    cfg = load_train_config(path.as_posix())
    assert cfg.temperature_split == "calibration"
