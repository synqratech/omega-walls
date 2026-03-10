from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def export_pitheta_adapter(*, checkpoint_dir: str, export_dir: str, model_manifest: str | None = None) -> Dict[str, str]:
    ckpt = Path(checkpoint_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint dir not found: {ckpt.as_posix()}")
    export = Path(export_dir)
    export.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(model_manifest) if model_manifest else (ckpt / "model_manifest.json")
    if not manifest_path.exists():
        raise FileNotFoundError(f"model manifest not found: {manifest_path.as_posix()}")

    shutil.copy2(manifest_path, export / "model_manifest.json")
    if (ckpt / "heads.pt").exists():
        shutil.copy2(ckpt / "heads.pt", export / "heads.pt")
    if (ckpt / "encoder_state.pt").exists():
        shutil.copy2(ckpt / "encoder_state.pt", export / "encoder_state.pt")
    if (ckpt / "temperature_scaling.json").exists():
        shutil.copy2(ckpt / "temperature_scaling.json", export / "temperature_scaling.json")
    _copy_tree(ckpt / "adapter", export / "adapter")
    _copy_tree(ckpt / "tokenizer", export / "tokenizer")

    runtime_cfg = {
        "checkpoint_dir": export.as_posix(),
        "heads": "heads.pt",
        "adapter_dir": "adapter",
        "tokenizer_dir": "tokenizer",
        "calibration_file": "temperature_scaling.json",
    }
    (export / "runtime_config.json").write_text(json.dumps(runtime_cfg, ensure_ascii=True, indent=2), encoding="utf-8")

    checksums: Dict[str, str] = {}
    for path in sorted(p for p in export.rglob("*") if p.is_file()):
        checksums[path.relative_to(export).as_posix()] = _sha256_file(path)
    (export / "checksums.json").write_text(json.dumps(checksums, ensure_ascii=True, indent=2), encoding="utf-8")
    return {
        "export_dir": export.as_posix(),
        "files": str(len(checksums)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export PiTheta adapter for runtime projector.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--model-manifest", default=None)
    args = parser.parse_args()

    report = export_pitheta_adapter(
        checkpoint_dir=str(args.checkpoint_dir),
        export_dir=str(args.export_dir),
        model_manifest=args.model_manifest,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
