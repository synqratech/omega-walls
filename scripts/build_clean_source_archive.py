"""Build a deterministic, secret-scanned Omega Walls source archive."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Sequence
import zipfile

try:
    from scripts.secret_scan import scan_tree
except ModuleNotFoundError:  # direct execution: python scripts/build_clean_source_archive.py
    from secret_scan import scan_tree


ROOT = Path(__file__).resolve().parent.parent
EXCLUDED_PARTS = {
    ".git", ".pytest_cache", ".ruff_cache", "__pycache__", "artifacts", "build",
    "dist", "node_modules", "omega_walls.egg-info", "tmp_codex_pytest", "_tmp",
}
EXCLUDED_NAMES = {".env", ".env.local", ".env.prod", ".env.production"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".pyd", ".log", ".zip", ".tar", ".gz"}
ALLOWED_RELEASE_PREFIXES = {
    "plugins/openclaw-omega-guard/dist/",
}
ALLOWED_RELEASE_ARTIFACTS = {
    "artifacts/vision_phase1/frozen/vision_phase1_frozen_v1.json",
    "artifacts/vision_wave_b/frozen/vision_wave_b_frozen_v1.json",
    "artifacts/vision_wave_b/local_rapidocr/vision_wave_b_local_rapidocr_v1.json",
    "artifacts/vision_wave_c/frozen/vision_wave_c_frozen_v1.json",
    "artifacts/vision_wave_c/local/vision_wave_c_local_v1.json",
}


def _included_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"source archive refuses symlink: {path.relative_to(root)}")
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        rel_posix = rel.as_posix()
        allow_excluded_path = any(
            rel_posix.startswith(prefix) for prefix in ALLOWED_RELEASE_PREFIXES
        )
        if (
            rel_posix not in ALLOWED_RELEASE_ARTIFACTS
            and not allow_excluded_path
            and any(part in EXCLUDED_PARTS for part in rel.parts)
        ):
            continue
        if path.name in EXCLUDED_NAMES or (path.name.startswith(".env.") and path.name != ".env.example"):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        yield path


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_archive(*, root: Path, output: Path, prefix: str = "OmegaWalls") -> dict:
    root = root.resolve()
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="omega-source-") as tmp:
        stage = Path(tmp) / prefix
        stage.mkdir(parents=True)
        manifest_files: list[dict[str, object]] = []
        for source in _included_files(root):
            rel = source.relative_to(root)
            target = stage / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest_files.append({
                "path": rel.as_posix(),
                "sha256": _sha256_file(target),
                "bytes": target.stat().st_size,
            })

        findings = scan_tree(stage)
        if findings:
            summary = ", ".join(f"{f.path}:{f.line}:{f.rule}" for f in findings[:10])
            raise RuntimeError(f"secret scan failed; archive not created: {summary}")

        manifest = {
            "schema_version": "1.0",
            "generator": "scripts/build_clean_source_archive.py",
            "secret_scan": "passed",
            "file_count": len(manifest_files),
            "files": manifest_files,
        }
        manifest_path = stage / "SOURCE_ARCHIVE_MANIFEST.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

        if output.exists():
            output.unlink()
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(stage.rglob("*")):
                if not path.is_file():
                    continue
                arcname = path.relative_to(stage.parent).as_posix()
                info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, path.read_bytes())

    return {
        "event": "omega_clean_source_archive_v1",
        "status": "ok",
        "output": str(output),
        "sha256": _sha256_file(output),
        "file_count": len(manifest_files) + 1,
        "secret_scan": "passed",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a deterministic secret-free source zip")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--output", default=str(ROOT / "dist" / "OmegaWalls-source.zip"))
    parser.add_argument("--prefix", default="OmegaWalls")
    args = parser.parse_args(argv)
    report = build_archive(root=Path(args.root), output=Path(args.output), prefix=str(args.prefix))
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
