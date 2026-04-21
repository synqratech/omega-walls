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
    "README.md",
    "docs/README.md",
    "docs/quickstart.md",
    "docs/framework_integrations_quickstart.md",
    "docs/custom_integration_from_scratch.md",
    "docs/monitoring_alerts.md",
    "docs/workflow_continuity.md",
    "docs/policy_tuning.md",
    "docs/config.md",
    "docs/tests_and_eval.md",
    "docs/benchmark_data_sources.md",
    "docs/architecture.md",
    "docs/threat_model.md",
    "docs/debugging_workflow_failures.md",
    "docs/openclaw_integration.md",
    "docs/release_surfaces.md",
]

MD_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
ABS_BLOB_PREFIX = "https://github.com/synqratech/omega-walls/blob/main/"


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


def validate(*, manifest: Path) -> dict:
    local_targets = _collect_local_targets()
    pypi_targets = _collect_pypi_blob_targets()
    required_targets = {p.resolve() for p in local_targets.union(pypi_targets)}

    with tempfile.TemporaryDirectory(prefix="oss_export_contract_", dir=str(ROOT / "artifacts")) as tmp_dir:
        export_dir = Path(tmp_dir) / "export"
        export_allowlist(
            root=ROOT,
            manifest_path=manifest,
            output_dir=export_dir,
            clean=True,
        )
        missing: List[str] = []
        for target in sorted(required_targets):
            rel = target.relative_to(ROOT)
            exported = (export_dir / rel).exists()
            if not exported:
                missing.append(rel.as_posix())

    if missing:
        raise RuntimeError(f"required linked files are not present in OSS export: {missing}")

    return {
        "event": "validate_oss_docs_contract_v1",
        "status": "ok",
        "checked_docs": CURATED_DOCS,
        "linked_targets_checked": len(required_targets),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OSS docs/readme links against curated export manifest.")
    parser.add_argument("--manifest", default="config/oss_export_github.json")
    args = parser.parse_args()

    report = validate(manifest=(ROOT / str(args.manifest)).resolve())
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
