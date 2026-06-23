from __future__ import annotations

from typing import Any, Dict


def validate_benchmark_configs(config: Dict[str, Any]) -> None:
    bipia_cfg = config.get("bipia", {})
    if bipia_cfg:
        mode = str(bipia_cfg.get("mode_default", "sampled")).lower()
        if mode not in {"sampled", "full"}:
            raise ValueError("bipia.mode_default must be sampled|full")
        split = str(bipia_cfg.get("split_default", "test")).lower()
        if split != "test":
            raise ValueError("bipia.split_default must be test in v1")
        sampled = bipia_cfg.get("sampled", {})
        max_contexts = int(sampled.get("max_contexts_per_task", 20))
        max_attacks = int(sampled.get("max_attacks_per_task", 10))
        if max_contexts <= 0:
            raise ValueError("bipia.sampled.max_contexts_per_task must be > 0")
        if max_attacks <= 0:
            raise ValueError("bipia.sampled.max_attacks_per_task must be > 0")
        thresholds = bipia_cfg.get("thresholds", {}).get("sampled", {})
        for key in ("attack_off_rate_ge", "per_task_attack_off_rate_ge", "coverage_wall_any_ge"):
            if float(thresholds.get(key, 0.0)) < 0.0:
                raise ValueError(f"bipia.thresholds.sampled.{key} must be >= 0")

    deepset_cfg = config.get("deepset", {})
    if deepset_cfg:
        mode = str(deepset_cfg.get("mode_default", "full")).lower()
        if mode not in {"sampled", "full"}:
            raise ValueError("deepset.mode_default must be sampled|full")
        split = str(deepset_cfg.get("split_default", "test")).lower()
        if split not in {"train", "test"}:
            raise ValueError("deepset.split_default must be train|test")
        label_attack = int(deepset_cfg.get("label_attack_value", 1))
        if label_attack not in {0, 1}:
            raise ValueError("deepset.label_attack_value must be 0|1")
        sampled = deepset_cfg.get("sampled", {}) or {}
        max_samples = int(sampled.get("max_samples", 116))
        if max_samples <= 0:
            raise ValueError("deepset.sampled.max_samples must be > 0")
        thresholds = (deepset_cfg.get("thresholds", {}) or {}).get("report", {}) or {}
        for key in ("attack_off_rate_ge", "coverage_wall_any_attack_ge", "f1_ge"):
            val = float(thresholds.get(key, 0.0))
            if val < 0.0 or val > 1.0:
                raise ValueError(f"deepset.thresholds.report.{key} must be in [0,1]")
        benign_off = float(thresholds.get("benign_off_rate_le", 1.0))
        if benign_off < 0.0 or benign_off > 1.0:
            raise ValueError("deepset.thresholds.report.benign_off_rate_le must be in [0,1]")
        repro = deepset_cfg.get("reproducibility", {}) or {}
        if int(repro.get("seed_default", 41)) < 0:
            raise ValueError("deepset.reproducibility.seed_default must be >= 0")

