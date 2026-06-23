from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from omega.release.manifest import artifact_record, build_release_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an Omega Enterprise release manifest")
    parser.add_argument("--artifact", action="append", default=[], help="legacy name:kind:path[:oci_digest]")
    parser.add_argument("--file-artifact", action="append", default=[], help="name=kind=path")
    parser.add_argument("--oci-artifact", action="append", default=[], help="name=reference=digest")
    parser.add_argument("--channel", choices=["candidate", "stable", "lts"], default=None)
    parser.add_argument("--git-commit", default=None)
    parser.add_argument("--build-id", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    artifacts = {}
    for raw in args.artifact:
        parts = str(raw).split(":", 3)
        if len(parts) < 3:
            raise ValueError("--artifact must be name:kind:path[:oci_digest]")
        name, kind, path = parts[:3]
        oci_digest = parts[3] if len(parts) == 4 else None
        artifacts[name] = artifact_record(path, kind=kind, oci_digest=oci_digest)
    for raw in args.file_artifact:
        name, kind, path = str(raw).split("=", 2)
        artifacts[name] = artifact_record(path, kind=kind)
    for raw in args.oci_artifact:
        name, reference, digest = str(raw).split("=", 2)
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError("--oci-artifact digest must be sha256:<64 hex>")
        if ":latest" in reference.lower():
            raise ValueError("mutable latest OCI reference is forbidden")
        artifacts[name] = {
            "kind": "oci_image",
            "path": f"{reference}@{digest}",
            "sha256": digest.removeprefix("sha256:"),
            "oci_digest": digest,
            "size_bytes": 0,
        }
    if not artifacts:
        raise ValueError("at least one --artifact is required")
    manifest = build_release_manifest(
        artifacts=artifacts,
        channel=args.channel,
        git_commit=args.git_commit,
        build_id=args.build_id,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(output), "release": manifest["release"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
