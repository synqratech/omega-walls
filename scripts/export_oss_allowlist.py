from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parent.parent


def _load_manifest(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    include = payload.get("include", [])
    exclude_globs = payload.get("exclude_globs", [])
    if not isinstance(include, list) or not isinstance(exclude_globs, list):
        raise ValueError("allowlist manifest must contain include[] and exclude_globs[] lists")
    return {
        "include": [str(x) for x in include if str(x).strip()],
        "exclude_globs": [str(x) for x in exclude_globs if str(x).strip()],
    }


def _is_excluded(rel_path: str, exclude_globs: Iterable[str]) -> bool:
    rel = rel_path.replace("\\", "/")
    for patt in exclude_globs:
        if fnmatch.fnmatch(rel, patt):
            return True
    return False


def _iter_files(base: Path) -> Iterable[Path]:
    if base.is_file():
        yield base
        return
    for item in base.rglob("*"):
        if item.is_file():
            yield item


def export_allowlist(
    *,
    root: Path,
    manifest_path: Path,
    output_dir: Path,
    clean: bool = False,
) -> Dict[str, Any]:
    cfg = _load_manifest(manifest_path)
    include = cfg["include"]
    exclude_globs = cfg["exclude_globs"]

    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: List[str] = []
    skipped: List[str] = []
    missing: List[str] = []

    for rel_item in include:
        src = (root / rel_item).resolve()
        if not src.exists():
            missing.append(rel_item)
            continue
        for src_file in _iter_files(src):
            rel = src_file.relative_to(root).as_posix()
            if _is_excluded(rel, exclude_globs):
                skipped.append(rel)
                continue
            dst = output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst)
            copied.append(rel)

    report = {
        "event": "oss_export_allowlist_v1",
        "root": str(root.as_posix()),
        "manifest": str(manifest_path.as_posix()),
        "output_dir": str(output_dir.as_posix()),
        "include_items": include,
        "exclude_globs": exclude_globs,
        "copied_total": int(len(copied)),
        "skipped_total": int(len(skipped)),
        "missing_total": int(len(missing)),
        "missing": sorted(set(missing)),
        "status": "ok",
    }
    report_path = output_dir / "export_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Export OSS-ready tree by allowlist manifest.")
    parser.add_argument("--manifest", default="config/oss_export_allowlist.json")
    parser.add_argument("--output-dir", default="artifacts/oss_export")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    report = export_allowlist(
        root=ROOT,
        manifest_path=(ROOT / str(args.manifest)).resolve(),
        output_dir=(ROOT / str(args.output_dir)).resolve(),
        clean=bool(args.clean),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
