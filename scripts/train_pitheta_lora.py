from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.pitheta.training import train_pitheta_lora


def main() -> int:
    parser = argparse.ArgumentParser(description="Train PiTheta LoRA checkpoint.")
    parser.add_argument("--train-config", default="config/pitheta_train.yml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--fit-temperature", choices=["true", "false"], default="true")
    parser.add_argument("--temperature-split", choices=["dev", "holdout", "calibration"], default="dev")
    parser.add_argument("--temperature-output", default="best/temperature_scaling.json")
    parser.add_argument("--calibration-source-mode", choices=["dataset", "gold_only", "blended"], default=None)
    parser.add_argument("--calibration-gold-slice-path", default=None)
    parser.add_argument("--calibration-gold-ratio", type=float, default=None)
    parser.add_argument("--calibration-weak-ratio", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    import yaml

    train_cfg = yaml.safe_load(Path(args.train_config).read_text(encoding="utf-8")) or {}
    inner = train_cfg.get("pitheta_train", train_cfg)
    if not isinstance(inner, dict):
        raise ValueError("pitheta_train config must be mapping")
    calibration = inner.get("calibration", {}) or {}
    calibration = {
        **calibration,
        "fit_temperature": str(args.fit_temperature).lower() == "true",
        "temperature_split": str(args.temperature_split),
        "temperature_output": str(args.temperature_output),
    }
    if args.calibration_source_mode is not None:
        calibration["source_mode"] = str(args.calibration_source_mode)
    if args.calibration_gold_slice_path is not None:
        calibration["gold_slice_path"] = str(args.calibration_gold_slice_path)
    if args.calibration_gold_ratio is not None:
        calibration["gold_ratio"] = float(args.calibration_gold_ratio)
    if args.calibration_weak_ratio is not None:
        calibration["weak_ratio"] = float(args.calibration_weak_ratio)
    inner["calibration"] = calibration
    if args.epochs is not None:
        inner["epochs"] = int(args.epochs)
    if "pitheta_train" in train_cfg:
        train_cfg["pitheta_train"] = inner
    else:
        train_cfg = inner

    effective_cfg_path = Path(args.output_dir) / "effective_train_config.yml"
    effective_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    effective_cfg_path.write_text(yaml.safe_dump(train_cfg, sort_keys=False), encoding="utf-8")

    manifest = train_pitheta_lora(
        train_config_path=str(effective_cfg_path),
        data_dir=str(args.data_dir),
        output_dir=str(args.output_dir),
        resume_from=args.resume_from,
    )
    print(json.dumps(manifest, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
