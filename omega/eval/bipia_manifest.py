"""BIPIA reproducibility and data-readiness manifests."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from omega.eval.bipia_adapter import iter_required_bipia_files


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_md5_pairs(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            continue
        out[parts[1]] = parts[0]
    return out


def compute_tree_sha256(root: str) -> str:
    base = Path(root)
    h = hashlib.sha256()
    for path in sorted(p for p in base.rglob("*") if p.is_file()):
        rel = path.relative_to(base).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        h.update(_sha256_file(path).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


def verify_qa_abstract_md5(benchmark_root: str) -> Tuple[bool, Dict[str, Any]]:
    root = Path(benchmark_root)
    checks: Dict[str, Dict[str, Any]] = {}
    ok = True
    for task in ("qa", "abstract"):
        md5_path = root / task / "md5.txt"
        expected = _load_md5_pairs(md5_path)
        task_checks: Dict[str, Any] = {}
        for name in ("train.jsonl", "test.jsonl"):
            file_path = root / task / name
            exists = file_path.exists()
            actual = _md5_file(file_path) if exists else None
            wanted = expected.get(name)
            file_ok = bool(exists and actual is not None and wanted is not None and actual == wanted)
            task_checks[name] = {
                "exists": exists,
                "expected_md5": wanted,
                "actual_md5": actual,
                "ok": file_ok,
            }
            ok = ok and file_ok
        checks[task] = task_checks
    return ok, checks


@dataclass(frozen=True)
class BIPIAManifestInput:
    benchmark_root: str
    split: str
    mode: str
    seed_pack: List[int]
    config_refs: Dict[str, str]
    commit_env_var: str = "BIPIA_COMMIT"
    strict: bool = True


def build_bipia_manifest(args: BIPIAManifestInput) -> Dict[str, Any]:
    root = Path(args.benchmark_root)
    missing_files: List[str] = []
    required_files = [str(p.as_posix()) for p in iter_required_bipia_files(args.benchmark_root, args.split)]
    file_hashes: Dict[str, str] = {}
    for file_path in required_files:
        path = Path(file_path)
        if not path.exists():
            missing_files.append(path.as_posix())
            continue
        file_hashes[path.as_posix()] = _sha256_file(path)

    qa_abstract_md5_ok, md5_checks = verify_qa_abstract_md5(args.benchmark_root)
    bipia_commit = os.getenv(args.commit_env_var, "").strip()
    if args.strict and not bipia_commit:
        raise ValueError(
            f"Missing required env {args.commit_env_var} for strict BIPIA manifest. "
            f"Set {args.commit_env_var}=<upstream commit sha>."
        )
    if args.strict and missing_files:
        raise ValueError(f"Missing required BIPIA files: {missing_files}")
    if args.strict and not qa_abstract_md5_ok:
        raise ValueError("qa/abstract md5 checks failed")

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_root": root.as_posix(),
        "split": args.split,
        "mode": args.mode,
        "seed_pack": list(args.seed_pack),
        "config_refs": dict(args.config_refs),
        "bipia_commit": bipia_commit if bipia_commit else "unknown",
        "bipia_commit_env_var": args.commit_env_var,
        "bipia_tree_sha256": compute_tree_sha256(root.as_posix()) if root.exists() else None,
        "required_files": required_files,
        "missing_files": missing_files,
        "file_hashes": file_hashes,
        "data_readiness": {
            "qa_abstract_md5_ok": qa_abstract_md5_ok,
            "qa_abstract_md5_checks": md5_checks,
        },
    }


def write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
