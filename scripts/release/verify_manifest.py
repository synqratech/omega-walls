from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.release.manifest import sha256_file, validate_release_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify an Omega Enterprise release manifest")
    parser.add_argument("manifest")
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    path = Path(args.manifest)
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_release_manifest(payload, check_runtime_metadata=False)
    verified_files = 0
    verified_oci = 0
    if args.verify_files:
        for name, artifact in payload["artifacts"].items():
            if artifact.get("kind") == "oci_image":
                digest = str(artifact.get("oci_digest", ""))
                if not digest.startswith("sha256:") or artifact.get("sha256") != digest.removeprefix("sha256:"):
                    raise ValueError(f"OCI artifact digest mismatch: {name}")
                if not str(artifact.get("path", "")).endswith("@" + digest):
                    raise ValueError(f"OCI artifact path is not digest-pinned: {name}")
                verified_oci += 1
                continue
            artifact_path = Path(str(artifact.get("path", "")))
            if not artifact_path.is_absolute():
                artifact_path = ROOT / artifact_path
            if not artifact_path.is_file():
                raise FileNotFoundError(f"artifact {name} not found: {artifact_path}")
            if sha256_file(artifact_path) != artifact["sha256"]:
                raise ValueError(f"artifact hash mismatch: {name}")
            verified_files += 1
    print(json.dumps({"status": "ok", "release": payload["release"], "verified_files": verified_files, "verified_oci": verified_oci}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
