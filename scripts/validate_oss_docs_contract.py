from __future__ import annotations

import argparse
import json
import re
import tempfile
from pathlib import Path
from typing import Iterable, List, Set

ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_oss_allowlist import export_allowlist

CURATED_DOCS: List[str] = [
    "README_OSS.md",
    "docs/README.md",
    "docs/quickstart.md",
    "docs/config.md",
    "docs/framework_integrations_quickstart.md",
]
CURATED_DOC_EXPORT_MAP = {
    "README_OSS.md": "README.md",
}

MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ABS_BLOB_PREFIX = "https://github.com/synqratech/omega-walls/blob/main/"
REQUIRED_EXPORT_EXCLUDES: Set[str] = {
    "config/profiles/prod_enterprise.yml",
    "config/profiles/prod_vision_enterprise.yml",
    "config/profiles/sensitive_hybrid_redacted.yml",
    "config/profiles/sensitive_local_semantic.yml",
    "docs/enterprise_pilot_guide.md",
    "docs/pilot_operations_runbook.md",
    "scripts/omega_walls_enterprise.py",
}


def _extract_links(text: str) -> Iterable[str]:
    for match in MD_LINK_RE.finditer(text):
        yield str(match.group(1)).strip()


def _collect_local_targets() -> Set[Path]:
    targets: Set[Path] = set()
    for rel in CURATED_DOCS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        for link in _extract_links(text):
            if not link or link.startswith("#"):
                continue
            low = link.lower()
            if low.startswith(("http://", "https://", "mailto:")):
                continue
            target = (path.parent / link.split("#", 1)[0]).resolve()
            if not target.exists():
                raise RuntimeError(f"broken local link in {rel}: {link}")
            targets.add(target)
    return targets


def _collect_pypi_blob_targets() -> Set[Path]:
    pypi = ROOT / "README_PYPI.md"
    text = pypi.read_text(encoding="utf-8")
    targets: Set[Path] = set()
    for line in text.splitlines():
        line = line.strip()
        if ABS_BLOB_PREFIX not in line:
            continue
        idx = line.find(ABS_BLOB_PREFIX)
        url = line[idx:].strip()
        rel = url.replace(ABS_BLOB_PREFIX, "").split(")", 1)[0]
        target = (ROOT / rel).resolve()
        if not target.exists():
            raise RuntimeError(f"broken README_PYPI absolute GitHub link target: {url}")
        targets.add(target)
    return targets


def _is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".omega_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _resolve_tmp_base(*, preferred: Path, fallback: Path) -> Path:
    if _is_writable_dir(preferred):
        return preferred
    if _is_writable_dir(fallback):
        return fallback
    raise RuntimeError("no writable temp base for OSS docs validation")


def _validate_manifest_contract(*, manifest: Path) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    include_entries = list(payload.get("include", []))
    include_sources = set()
    include_targets = set()
    for item in include_entries:
        if isinstance(item, str):
            include_sources.add(item)
            include_targets.add(item)
        elif isinstance(item, dict):
            src = str(item.get("src", "")).strip()
            dst = str(item.get("dst", "")).strip()
            if src:
                include_sources.add(src)
            if dst:
                include_targets.add(dst)
    excludes = {str(x) for x in list(payload.get("exclude_globs", []))}

    required_docs = {
        "README.md",
        "docs/README.md",
        "docs/quickstart.md",
        "docs/config.md",
        "docs/framework_integrations_quickstart.md",
    }
    missing_include = sorted(x for x in required_docs if x not in include_targets)
    missing_excludes = sorted(x for x in REQUIRED_EXPORT_EXCLUDES if x not in excludes)
    if "README_OSS.md" not in include_sources:
        missing_include.append("README_OSS.md (source)")
    if missing_include or missing_excludes:
        raise RuntimeError(
            f"manifest contract failed: missing_include={missing_include}, missing_excludes={missing_excludes}"
        )
    return {
        "required_docs": sorted(required_docs),
        "required_excludes": sorted(REQUIRED_EXPORT_EXCLUDES),
    }


def validate(*, manifest: Path, full_export: bool = False, tmp_base: Path | None = None) -> dict:
    local_targets = _collect_local_targets()
    pypi_targets = _collect_pypi_blob_targets()
    required_targets = {p.resolve() for p in local_targets.union(pypi_targets)}
    manifest_contract = _validate_manifest_contract(manifest=manifest)
    mode = "fast_manifest_link_check"
    missing: List[str] = []

    if full_export:
        preferred_tmp = Path("C:/tmp")
        fallback_tmp = ROOT / "artifacts"
        resolved_tmp_base = _resolve_tmp_base(
            preferred=(Path(tmp_base).resolve() if tmp_base is not None else preferred_tmp),
            fallback=fallback_tmp,
        )
        mode = "full_export_check"
        with tempfile.TemporaryDirectory(prefix="oss_export_contract_", dir=str(resolved_tmp_base)) as tmp_dir:
            export_dir = Path(tmp_dir) / "export"
            export_allowlist(
                root=ROOT,
                manifest_path=manifest,
                output_dir=export_dir,
                clean=True,
            )
            for target in sorted(required_targets):
                rel = target.relative_to(ROOT).as_posix()
                export_rel = CURATED_DOC_EXPORT_MAP.get(rel, rel)
                exported = (export_dir / export_rel).exists()
                if not exported:
                    missing.append(export_rel)
        if missing:
            raise RuntimeError(f"required linked files are not present in OSS export: {missing}")

    return {
        "event": "validate_oss_docs_contract_v1",
        "status": "ok",
        "mode": mode,
        "checked_docs": CURATED_DOCS,
        "linked_targets_checked": len(required_targets),
        "manifest_contract": manifest_contract,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OSS docs/readme links against curated export manifest.")
    parser.add_argument("--manifest", default="config/oss_export_github.json")
    parser.add_argument("--full-export", action="store_true")
    parser.add_argument("--tmp-base", default=None)
    args = parser.parse_args()

    report = validate(
        manifest=(ROOT / str(args.manifest)).resolve(),
        full_export=bool(args.full_export),
        tmp_base=(Path(str(args.tmp_base)).resolve() if args.tmp_base else None),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
