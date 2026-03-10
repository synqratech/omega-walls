"""Deepset reproducibility manifest helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from omega.eval.bipia_manifest import compute_tree_sha256
from omega.eval.deepset_adapter import deepset_split_stats, iter_required_deepset_files


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


@dataclass(frozen=True)
class DeepsetManifestInput:
    benchmark_root: str
    split: str
    mode: str
    seed_pack: List[int]
    label_attack_value: int
    config_refs: Dict[str, str]
    strict: bool = True


def build_deepset_manifest(args: DeepsetManifestInput) -> Dict[str, Any]:
    root = Path(args.benchmark_root)
    required_files = [str(p.as_posix()) for p in iter_required_deepset_files(args.benchmark_root)]
    missing_files: List[str] = []
    file_hashes: Dict[str, str] = {}

    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            missing_files.append(path.as_posix())
            continue
        file_hashes[path.as_posix()] = _sha256_file(path)

    train_rows, train_labels = deepset_split_stats(args.benchmark_root, "train")
    test_rows, test_labels = deepset_split_stats(args.benchmark_root, "test")

    if args.strict and missing_files:
        raise ValueError(f"Missing required deepset files: {missing_files}")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": root.as_posix(),
        "split": args.split,
        "mode": args.mode,
        "seed_pack": list(args.seed_pack),
        "label_attack_value": int(args.label_attack_value),
        "config_refs": dict(args.config_refs),
        "deepset_tree_sha256": compute_tree_sha256(root.as_posix()) if root.exists() else None,
        "required_files": required_files,
        "missing_files": missing_files,
        "file_hashes": file_hashes,
        "data_readiness": {
            "schema": ["text", "label"],
            "rows": {
                "train": train_rows,
                "test": test_rows,
            },
            "label_distribution": {
                "train": {str(k): v for k, v in sorted(train_labels.items())},
                "test": {str(k): v for k, v in sorted(test_labels.items())},
            },
        },
    }


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
